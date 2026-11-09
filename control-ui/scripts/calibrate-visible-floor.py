#!/usr/bin/env python3
"""Refine a Marble world's visible floor without changing its metric scale.

Run after calibrate-factory-world.py. Requires NumPy; uses local assets only.
The full-resolution SPZ floor is fitted around the workcell. A proper rigid
rotation/translation levels that surface at z=0 before the renderer applies
its native/offline floor offset. The API uniform scale is preserved exactly.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


def transform_hash(matrix: np.ndarray) -> str:
    return hashlib.sha256(json.dumps(matrix.tolist(), separators=(",", ":")).encode()).hexdigest()


def read_floor_splats(path: Path, metric_scale: float, max_size_m: float) -> tuple[np.ndarray, dict]:
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if len(data) < 16:
        raise ValueError("SPZ is too short for a header")
    magic, version, count, degree, bits, flags, _ = struct.unpack_from("<IIIBBBB", data)
    if magic != 0x5053474E or version not in (2, 3):
        raise ValueError("Expected SPZ v2/v3 positions")
    rotation_width = 3 if version == 2 else 4
    expected = 16 + count * (16 + rotation_width + 3 * ((degree + 1) ** 2 - 1))
    if len(data) < expected:
        raise ValueError("SPZ is truncated")
    alpha = np.frombuffer(data, dtype=np.uint8, count=count, offset=16 + count * 9)
    scales = np.frombuffer(data, dtype=np.uint8, count=count * 3, offset=16 + count * 13).reshape(-1, 3)
    largest_size = np.exp(scales.astype(np.float64) / 16 - 10).max(axis=1) * metric_scale
    keep = (alpha > 127.5) & (largest_size < max_size_m)
    packed = np.frombuffer(data, dtype=np.uint8, count=count * 9, offset=16).reshape(-1, 3, 3)[keep].astype(np.int32)
    fixed = packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    points = fixed.astype(np.float64) / (1 << bits)
    return points, {"file": path.name, "version": version, "totalSplats": count,
                    "eligibleSplats": int(keep.sum()), "minimumOpacity": 0.5,
                    "maximumGaussianSizeM": max_size_m, "fileBytes": path.stat().st_size}


def load_base(directory: Path, manifest: dict) -> tuple[np.ndarray, str]:
    current = np.array(manifest.get("calibration", {}).get("rawToWorld"), dtype=np.float64)
    if current.shape != (4, 4) or not np.isfinite(current).all():
        raise ValueError("Run the metric/collider calibrator first: valid rawToWorld is required")
    provenance = manifest.get("calibration", {}).get("visibleFloorProvenance")
    if provenance:
        report_path = directory / "visible-floor-report.json"
        if not report_path.exists():
            raise ValueError("Previous visible-floor refinement is declared but its report is missing; cannot safely refine twice")
        previous = json.loads(report_path.read_text())
        world_id = manifest.get("worldId") or manifest.get("world_id")
        if previous.get("worldId") != world_id:
            raise ValueError("Previous visible-floor report belongs to another world")
        refined = np.array(previous.get("refinedRawToWorld"), dtype=np.float64)
        base = np.array(previous.get("baseRawToWorld"), dtype=np.float64)
        if refined.shape == (4, 4) and base.shape == (4, 4) and np.allclose(current, refined, rtol=0, atol=1e-10):
            return base, "original_base_from_previous_visible_floor_report"
        # The metric/collider calibrator may have reset rawToWorld since the last run.
        # In that case the current matrix, rather than the previous refinement, is the new base.
    return current, "current_metric_collider_calibration"


def cell_medians(points: np.ndarray, cell_m: float) -> np.ndarray:
    keys = np.floor(points[:, :2] / cell_m).astype(np.int64)
    _, indices, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(indices)
    offsets = np.r_[0, np.cumsum(counts)]
    return np.array([np.median(points[order[offsets[i]:offsets[i + 1]]], axis=0)
                     for i in range(len(counts)) if counts[i] >= 4])


def fit_visible_plane(cells: np.ndarray, max_tilt_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic plane consensus, then robust least-squares refits of cell medians."""
    if len(cells) < 60:
        raise ValueError(f"Insufficient visible floor coverage: only {len(cells)} populated cells (need 60)")
    design = np.column_stack([cells[:, :2], np.ones(len(cells))])
    rng = np.random.default_rng(13)
    best, best_count, best_residual = None, 0, math.inf
    for _ in range(192):
        sample = rng.choice(len(cells), 3, replace=False)
        try:
            candidate = np.linalg.solve(design[sample], cells[sample, 2])
        except np.linalg.LinAlgError:
            continue
        if math.degrees(math.atan(float(np.linalg.norm(candidate[:2])))) > max_tilt_degrees:
            continue
        residual = np.abs(cells[:, 2] - np.einsum("ni,i->n", design, candidate))
        keep = residual < 0.02
        count = int(keep.sum())
        score = float(np.median(residual[keep])) if count else math.inf
        if count > best_count or (count == best_count and score < best_residual):
            best, best_count, best_residual = keep, count, score
    if best is None or best_count < 60:
        raise ValueError("No coherent near-horizontal visible floor plane with at least 60 cells")
    keep = best
    for _ in range(5):
        coefficient = np.linalg.lstsq(design[keep], cells[keep, 2], rcond=None)[0]
        residual = cells[:, 2] - np.einsum("ni,i->n", design, coefficient)
        threshold = max(0.006, 3 * float(np.median(np.abs(residual[keep]))))
        keep = np.abs(residual) <= threshold
        if keep.sum() < 60:
            raise ValueError("Visible floor fit lost too much support during robust refitting")
    coefficient = np.linalg.lstsq(design[keep], cells[keep, 2], rcond=None)[0]
    return coefficient, keep


def rigid_floor_correction(coefficient: np.ndarray) -> np.ndarray:
    a, b, intercept = coefficient
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, target)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    # Rodrigues' rotation between two unit vectors. Their dot product is near +1.
    rotation = np.eye(3) + skew + np.einsum("ij,jk->ik", skew, skew) / (1 + float(normal[2]))
    correction = np.eye(4)
    correction[:3, :3] = rotation
    correction[:3, 3] = -np.einsum("ij,j->i", rotation, np.array([0.0, 0.0, intercept]))
    return correction


def refine(directory: Path, *, center_x: float = 0.25, center_y: float = 0.05,
           half_width: float = 1.0, half_length: float = 1.15, cell_m: float = 0.1,
           max_size_m: float = 0.05, max_tilt: float = 3.0) -> dict:
    manifest = json.loads((directory / "world.json").read_text())
    base, base_source = load_base(directory, manifest)
    lengths = np.linalg.norm(base[:3, :3], axis=0)
    scale = float(lengths.mean())
    rotation = base[:3, :3] / scale
    if scale <= 0 or not np.allclose(lengths, scale, rtol=1e-8) or not np.allclose(
            np.einsum("ji,jk->ik", rotation, rotation), np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError("Base matrix must contain a proper rotation and uniform positive metric scale")
    full_file = (manifest.get("splats") or {}).get("full")
    if not full_file:
        raise ValueError("The world package must provide its original full-resolution SPZ")
    points, splat_info = read_floor_splats(directory / full_file, scale, max_size_m)
    world = np.einsum("ij,nj->ni", base[:3, :3], points) + base[:3, 3]
    local = world[np.isfinite(world).all(axis=1)
                  & (np.abs(world[:, 0] - center_x) <= half_width)
                  & (np.abs(world[:, 1] - center_y) <= half_length)
                  & (world[:, 2] > -0.30) & (world[:, 2] < 0.25)]
    cells = cell_medians(local, cell_m)
    coefficient, keep = fit_visible_plane(cells, max_tilt)
    inliers = cells[keep]
    prediction = np.einsum("ni,i->n", np.column_stack([inliers[:, :2], np.ones(len(inliers))]), coefficient)
    residual = np.abs(inliers[:, 2] - prediction)
    median, p95 = map(float, np.percentile(residual, [50, 95]))
    tilt = math.degrees(math.atan(float(np.linalg.norm(coefficient[:2]))))
    spans = np.ptp(inliers[:, :2], axis=0)
    coverage = spans / np.array([2 * half_width, 2 * half_length])
    centered = inliers[:, :2] - np.array([center_x, center_y])
    _, quadrant_counts = np.unique((centered > 0).astype(int), axis=0, return_counts=True)
    quadrants = int(np.sum(quadrant_counts >= 5))
    errors, warnings = [], []
    if len(inliers) / len(cells) < 0.5:
        errors.append("Less than half the floor cells support one visible floor plane")
    if min(coverage) < 0.5 or quadrants < 3:
        errors.append("Visible floor support does not span enough of the workcell footprint")
    if tilt > max_tilt:
        errors.append(f"Additional visible-floor tilt {tilt:.3f} exceeds {max_tilt:g} degrees")
    if abs(float(coefficient[2])) > 0.25:
        errors.append("Visible floor is over 25 cm from the metric/collider floor; inspect before adopting")
    if median > 0.025 or p95 > 0.075:
        errors.append("Visible floor is too irregular (median >25 mm or P95 >75 mm)")
    elif median > 0.01 or p95 > 0.03:
        warnings.append("Visible floor residual exceeds the preferred 10 mm median / 30 mm P95")
    correction = rigid_floor_correction(coefficient)
    refined = np.einsum("ij,jk->ik", correction, base)
    corrected = np.einsum("ij,nj->ni", correction[:3, :3], inliers) + correction[:3, 3]
    refined_lengths = np.linalg.norm(refined[:3, :3], axis=0)
    scale_preserved = bool(np.allclose(refined_lengths, scale, rtol=1e-10, atol=1e-10))
    proper = bool(np.isclose(np.linalg.det(correction[:3, :3]), 1, atol=1e-10))
    if not scale_preserved or not proper:
        errors.append("Rigid correction did not preserve uniform scale / proper rotation")
    return {
        "version": 1, "worldId": manifest.get("worldId") or manifest.get("world_id"),
        "method": "local_full_resolution_visible_floor_cell_medians",
        "baseSource": base_source, "baseRawToWorld": base.tolist(),
        "baseTransformSha256": transform_hash(base),
        "visibleFloorCorrection": correction.tolist(), "refinedRawToWorld": refined.tolist(),
        "refinedTransformSha256": transform_hash(refined),
        "metricScalePreserved": scale_preserved, "properRotation": proper,
        "metersPerRawUnitBefore": scale, "metersPerRawUnitAfter": refined_lengths.tolist(),
        "spz": splat_info, "selectedSplats": len(local), "populatedCells": len(cells),
        "inlierCells": len(inliers), "inlierFraction": len(inliers) / len(cells),
        "cellSizeM": cell_m, "footprint": {"centerM": [center_x, center_y],
        "halfExtentsM": [half_width, half_length], "zBandM": [-0.30, 0.25]},
        "coverageFractionXY": coverage.tolist(), "coveredQuadrants": quadrants,
        "planeEquation": "z = a*x + b*y + c in the base calibrated world",
        "planeCoefficients": coefficient.tolist(), "additionalTiltDegrees": tilt,
        "interceptBeforeM": float(coefficient[2]),
        "fitMedianResidualM": median, "fitP95ResidualM": p95,
        "correctedFloorMedianAbsZM": float(np.median(np.abs(corrected[:, 2]))),
        "correctedFloorP95AbsZM": float(np.percentile(np.abs(corrected[:, 2]), 95)),
        "validation": {"status": "fail" if errors else "warn" if warnings else "pass",
                       "errors": errors, "warnings": warnings},
        "note": "This aligns visible Gaussian centers, not measured building dimensions. The API metric scale and robot/native poses remain unchanged; the renderer still applies its own floor offset.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "public/worlds/factory-conveyor")
    parser.add_argument("--center-x", type=float, default=0.25)
    parser.add_argument("--center-y", type=float, default=0.05)
    parser.add_argument("--half-width-m", type=float, default=1.0)
    parser.add_argument("--half-length-m", type=float, default=1.15)
    args = parser.parse_args()
    if not all(math.isfinite(v) for v in [args.center_x, args.center_y, args.half_width_m, args.half_length_m]) or min(args.half_width_m, args.half_length_m) <= 0:
        parser.error("Footprint coordinates must be finite and half extents positive")
    report = refine(args.directory, center_x=args.center_x, center_y=args.center_y,
                    half_width=args.half_width_m, half_length=args.half_length_m)
    report_file = args.directory / "visible-floor-report.json"
    if report["validation"]["errors"]:
        # Do not overwrite a previously applied report: it is needed to recover its original base.
        failure_file = args.directory / "visible-floor-rejected.json"
        failure_file.write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit("Visible floor refinement rejected; manifest unchanged. " + "; ".join(report["validation"]["errors"]))
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    manifest_file = args.directory / "world.json"
    manifest = json.loads(manifest_file.read_text())
    calibration = manifest.setdefault("calibration", {})
    calibration["rawToWorld"] = report["refinedRawToWorld"]
    calibration["visibleFloorProvenance"] = {
        "version": 1, "report": report_file.name, "method": report["method"],
        "baseTransformSha256": report["baseTransformSha256"],
        "refinedTransformSha256": report["refinedTransformSha256"],
        "metricScalePreserved": report["metricScalePreserved"],
        "additionalTiltDegrees": report["additionalTiltDegrees"],
        "interceptBeforeM": report["interceptBeforeM"],
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["worldId", "validation", "metricScalePreserved",
        "metersPerRawUnitBefore", "interceptBeforeM", "additionalTiltDegrees", "inlierCells",
        "correctedFloorMedianAbsZM", "correctedFloorP95AbsZM"]}, indent=2))


if __name__ == "__main__":
    main()
