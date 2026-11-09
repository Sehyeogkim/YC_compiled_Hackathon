#!/usr/bin/env python3
"""Calibrate Marble scale and floor offline, with metric metadata when available.

Requires Python 3 and NumPy. Standard exports use their API metric_scale_factor;
Draft exports fall back to an explicitly assumed capture height. A fitted floor
supplies tilt and alignment in both cases. API metric scale remains a model
estimate, not a measurement of the photographed building. Robot dimensions and
physics are never modified. No API calls or generation credits are used.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
from pathlib import Path

import numpy as np


def read_spz(path: Path) -> tuple[np.ndarray, dict]:
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    magic, version, count, degree, bits, flags, _ = struct.unpack_from("<IIIBBBB", data)
    if magic != 0x5053474E or version not in (2, 3):
        raise ValueError("Expected SPZ v2/v3 with 24-bit fixed-point positions")
    rotation_bytes = 3 if version == 2 else 4
    expected = 16 + count * (16 + rotation_bytes + 3 * ((degree + 1) ** 2 - 1))
    if len(data) < expected:
        raise ValueError(f"Truncated SPZ: {len(data)} bytes, expected {expected}")
    packed = np.frombuffer(data, dtype=np.uint8, count=count * 9, offset=16).reshape(-1, 3, 3).astype(np.int32)
    fixed = packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    points = fixed.astype(np.float64) / (1 << bits)
    alpha = np.frombuffer(data, dtype=np.uint8, count=count, offset=16 + count * 9)
    return points[alpha > 128], {"version": version, "count": count, "opaqueCount": int(np.sum(alpha > 128))}


def quaternion_matrix(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def read_collider(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read embedded, uncompressed glTF triangle geometry and node transforms."""
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data)
    if magic != 0x46546C67 or version != 2 or length != len(data):
        raise ValueError("Invalid GLB v2")
    offset, document, binary = 12, None, None
    while offset < length:
        size, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8:offset + 8 + size]
        if kind == 0x4E4F534A:
            document = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += size + 8
    if document is None or binary is None:
        raise ValueError("GLB needs JSON and embedded binary")

    def accessor(index: int) -> np.ndarray:
        a = document["accessors"][index]
        if "sparse" in a or "bufferView" not in a:
            raise ValueError("Sparse/implicit GLB accessors are unsupported")
        view = document["bufferViews"][a["bufferView"]]
        if view.get("buffer", 0) != 0 or view.get("extensions"):
            raise ValueError("External/compressed GLB buffers are unsupported")
        dtype = np.dtype({5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}[a["componentType"]])
        width = {"SCALAR": 1, "VEC3": 3}[a["type"]]
        start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride", width * dtype.itemsize)
        return np.ndarray((a["count"], width), dtype=dtype, buffer=binary,
                          offset=start, strides=(stride, dtype.itemsize)).copy()

    nodes = document.get("nodes", [])
    roots = document["scenes"][document.get("scene", 0)].get("nodes", [])
    stack = [(i, np.eye(4)) for i in roots]
    positions, triangles, vertex_offset = [], [], 0
    while stack:
        index, parent = stack.pop()
        node = nodes[index]
        if "matrix" in node:
            local = np.array(node["matrix"]).reshape(4, 4).T
        else:
            local = np.eye(4)
            local[:3, :3] = quaternion_matrix(node.get("rotation", [0, 0, 0, 1])) * node.get("scale", [1, 1, 1])
            local[:3, 3] = node.get("translation", [0, 0, 0])
        world = np.einsum("ij,jk->ik", parent, local)
        stack.extend((i, world) for i in node.get("children", []))
        if "mesh" not in node:
            continue
        for primitive in document["meshes"][node["mesh"]]["primitives"]:
            if primitive.get("mode", 4) != 4 or primitive.get("extensions"):
                raise ValueError("Expected uncompressed triangle primitives")
            p = accessor(primitive["attributes"]["POSITION"]).astype(np.float64)
            positions.append(np.einsum("ij,nj->ni", world[:3, :3], p) + world[:3, 3])
            indices = accessor(primitive["indices"]).ravel() if "indices" in primitive else np.arange(len(p))
            triangles.append(indices.astype(np.int64).reshape(-1, 3) + vertex_offset)
            vertex_offset += len(p)
    return np.vstack(positions), np.vstack(triangles)


def fit_floor(points: np.ndarray, initial_radius: float = 2.0) -> dict:
    """WEFARM-style lowest surface per horizontal cell, followed by robust SVD."""
    radius = initial_radius
    for _ in range(6):
        cell = radius / 40
        nearby = points[(np.hypot(points[:, 0], points[:, 2]) <= radius) & (points[:, 1] > 0)]
        if len(nearby) < 100:
            raise ValueError("Insufficient floor samples beneath the capture camera")
        keys = np.floor(nearby[:, [0, 2]] / cell).astype(np.int64)
        order = np.lexsort((-nearby[:, 1], keys[:, 1], keys[:, 0]))
        sorted_keys = keys[order]
        first = np.r_[True, np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)]
        surface = nearby[order][first]
        inliers = surface
        for _ in range(4):
            center = inliers.mean(axis=0)
            _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
            normal = vh[-1]
            if normal[1] > 0:
                normal = -normal
            distances = np.einsum("ni,i->n", surface - center, normal)
            threshold = max(3 * float(np.median(np.abs(distances))), 0.002)
            inliers = surface[np.abs(distances) <= threshold]
            if len(inliers) < 30:
                raise ValueError("Ground fit has too few inliers")
        height = float(-np.dot(normal, center))
        if height <= 0 or -normal[1] < 0.95:
            raise ValueError("Fitted surface is not a horizontal floor below the camera")
        residuals = np.abs(np.einsum("ni,i->n", inliers - center, normal))
        result = {"upNormalRaw": normal.tolist(), "cameraHeightRaw": height,
                  "planeEquation": "dot(upNormalRaw, rawPoint) + cameraHeightRaw = 0",
                  "tiltDegrees": math.degrees(math.acos(float(np.clip(-normal[1], -1, 1)))),
                  "surfaceCells": len(surface), "inlierCells": len(inliers),
                  "inlierFraction": len(inliers) / len(surface),
                  "horizontalSpanRaw": np.ptp(inliers[:, [0, 2]], axis=0).tolist(),
                  "coveredQuadrants": int(len(set(map(tuple, (inliers[:, [0, 2]] > 0).astype(int))))),
                  "medianResidualRaw": float(np.median(residuals)),
                  "p95ResidualRaw": float(np.percentile(residuals, 95)), "radiusRaw": radius}
        if abs(height * 4 - radius) < 0.01 * radius:
            return result
        radius = height * 4
    return result


def world_context(directory: Path) -> tuple[dict, dict, str]:
    """Read public metadata first; only match a private cached record to this world."""
    manifest_file = directory / "world.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    semantics = manifest.get("semantics")
    if isinstance(semantics, dict):
        return manifest, semantics, "world.json.semantics"
    # Legacy/private records may carry the original API assets shape.
    semantics = (manifest.get("assets") or {}).get("splats", {}).get("semantics_metadata")
    if isinstance(semantics, dict):
        return manifest, semantics, "world.json.assets.splats.semantics_metadata"
    cache_file = Path(__file__).resolve().parents[2] / ".marble/factory-standard-world.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        cached = cached.get("world", cached)
        public_id = manifest.get("worldId") or manifest.get("world_id")
        cached_id = cached.get("world_id") or cached.get("id")
        matches = public_id and public_id == cached_id
        if matches or (not manifest and directory.name == "factory-standard"):
            semantics = ((cached.get("assets") or {}).get("splats") or {}).get("semantics_metadata")
            if isinstance(semantics, dict):
                return manifest, semantics, "matching_cached_world.assets.splats.semantics_metadata"
    return manifest, {}, "absent"


def metric_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def calibrate(directory: Path, camera_height_m: float, camera_height_known: bool = False) -> dict:
    manifest, semantics, metadata_source = world_context(directory)
    splat_file = (manifest.get("splats") or {}).get("500k", "factory-500k.spz")
    points, header = read_spz(directory / splat_file)
    splat_fit = fit_floor(points)
    fits = {"splat": splat_fit}
    chosen, floor_source = splat_fit, "opaque_splat_floor"
    warnings, errors = [], []
    cross_check = {}
    collider_file = directory / "collider.glb"
    if collider_file.exists():
        vertices, triangles = read_collider(collider_file)
        reference = points[::max(1, len(points) // 16000)]
        sample = vertices[::max(1, len(vertices) // 180)]
        frame_scores = {}
        for name, signs in [("opencv", [1, 1, 1]), ("opengl", [1, -1, -1])]:
            distances = [float(np.min(np.linalg.norm(reference - p, axis=1))) for p in sample * signs]
            frame_scores[name] = float(np.median(distances))
        frame = min(frame_scores, key=frame_scores.get)
        sorted_scores = sorted(frame_scores.values())
        separation = (sorted_scores[1] - sorted_scores[0]) / max(sorted_scores[1], 1e-12)
        if separation < 0.10:
            errors.append(f"Collider coordinate frame is ambiguous (relative score separation {separation:.1%}).")
        elif separation < 0.25:
            warnings.append(f"Collider coordinate-frame evidence is weak (score separation {separation:.1%}).")
        vertices *= [1, 1, 1] if frame == "opencv" else [1, -1, -1]
        rng = np.random.default_rng(7)
        weights = rng.dirichlet([1, 1, 1], size=(len(triangles), 4))
        surface = np.einsum("tsk,tkd->tsd", weights, vertices[triangles]).reshape(-1, 3)
        collider_fit = fit_floor(np.vstack([vertices, surface]))
        fits["collider"] = {**collider_fit, "fileFrame": frame, "frameScoresRaw": frame_scores,
                            "frameScoreSeparation": separation}
        disagreement = abs(collider_fit["cameraHeightRaw"] / splat_fit["cameraHeightRaw"] - 1)
        normal_angle = math.degrees(math.acos(float(np.clip(np.dot(
            collider_fit["upNormalRaw"], splat_fit["upNormalRaw"]), -1, 1))))
        cross_check = {"heightRelativeDifference": disagreement, "normalAngleDegrees": normal_angle}
        if disagreement > 0.25:
            errors.append(f"Collider and splat floor heights differ by {disagreement:.0%} (>25%).")
        elif disagreement > 0.10:
            warnings.append(f"Collider and splat floor heights differ by {disagreement:.0%}; inspect alignment.")
        if normal_angle > 12:
            errors.append(f"Collider and splat floor normals differ by {normal_angle:.1f} degrees (>12).")
        elif normal_angle > 3:
            warnings.append(f"Collider and splat floor normals differ by {normal_angle:.1f} degrees.")
        chosen, floor_source = collider_fit, "collider_floor_cross_checked_against_splats"
    else:
        warnings.append("No collider available; fitted only the opaque splat floor.")

    assumed_scale = camera_height_m / chosen["cameraHeightRaw"]
    api_scale = metric_number(semantics.get("metric_scale_factor"))
    scale = api_scale if api_scale is not None else assumed_scale
    scale_source = "api_metric_scale_factor" if api_scale is not None else "assumed_camera_height"
    capture_height_m = scale * chosen["cameraHeightRaw"]
    scale_difference = abs(scale / assumed_scale - 1)
    if api_scale is not None and scale_difference > 0.15:
        qualification = "known" if camera_height_known else "assumed (unverified)"
        warnings.append(f"API scale implies {capture_height_m:.3f} m capture height, differing by "
                        f"{scale_difference:.0%} from the {qualification} {camera_height_m:g} m reference; "
                        "API scale retained, not silently replaced.")
        if camera_height_known and scale_difference > 0.50:
            errors.append("API scale differs by more than 50% from a declared known capture height.")
    if api_scale is None:
        if "metric_scale_factor" in semantics:
            warnings.append("Invalid API metric_scale_factor; using the declared camera-height assumption.")
        warnings.append("Absolute background scale is estimated from camera height, not supplied by the API.")
    checks = {
        "minimumInlierCells": (chosen["inlierCells"], 100, 40),
        "minimumInlierFraction": (chosen["inlierFraction"], 0.30, 0.15),
    }
    for name, (actual, warning_limit, failure_limit) in checks.items():
        if actual < failure_limit:
            errors.append(f"Floor coverage {name}: {actual:g} below minimum {failure_limit:g}.")
        elif actual < warning_limit:
            warnings.append(f"Floor coverage {name}: {actual:g} below preferred {warning_limit:g}.")
    span_fraction = min(chosen["horizontalSpanRaw"]) / chosen["radiusRaw"]
    if span_fraction < 0.10:
        errors.append("Floor inliers do not span enough area in both horizontal directions.")
    elif span_fraction < 0.40 or chosen["coveredQuadrants"] < 3:
        warnings.append("Floor coverage is spatially limited near the capture position.")
    for name, actual, warning_limit, failure_limit in [
        ("tiltDegrees", chosen["tiltDegrees"], 5, 15),
        ("medianResidualM", scale * chosen["medianResidualRaw"], 0.05, 0.20),
        ("p95ResidualM", scale * chosen["p95ResidualRaw"], 0.15, 0.50),
    ]:
        if actual > failure_limit:
            errors.append(f"Floor {name}: {actual:.4f} exceeds gross-error threshold {failure_limit:g}.")
        elif actual > warning_limit:
            warnings.append(f"Floor {name}: {actual:.4f} exceeds preferred {warning_limit:g}; inspect visually.")

    up = np.array(chosen["upNormalRaw"])
    forward = np.array([0.0, 0.0, 1.0])
    forward -= np.dot(forward, up) * up
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    rotation = np.vstack([right, forward, up])
    assert np.isclose(np.linalg.det(rotation), 1.0), "Must use a proper right-handed rotation"
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    # Use the independently fitted floor, never mix API offset with the fitted plane.
    matrix[2, 3] = capture_height_m
    transformed = np.einsum("ij,nj->ni", matrix[:3, :3], points) + matrix[:3, 3]
    if api_scale is not None:
        note = ("Uses API metric_scale_factor with independently fitted floor tilt/alignment. "
                "API metric scale is itself a model estimate, not measured architectural ground truth. "
                "The camera-height comparison does not override metric metadata.")
    else:
        note = (f"No valid API metric metadata. Scale assumes a {camera_height_m:g} m capture height; "
                "generated architecture is not measured ground truth.")
    report = {
        "version": 2, "method": floor_source, "scaleSource": scale_source,
        "isMetricScaleEstimated": api_scale is None,
        "metricScaleIsMeasuredGroundTruth": False,
        "assumedCameraHeightM": camera_height_m, "cameraHeightReferenceKnown": camera_height_known,
        "captureCameraHeightM": capture_height_m, "scaleNote": note,
        "estimatedMetersPerRawUnit": scale, "metersPerRawUnit": scale,
        "scaleMetadataSource": metadata_source, "apiSemanticsMetadata": semantics or None,
        "scaleCrossCheck": {"apiScale": api_scale, "cameraHeightAlternativeScale": assumed_scale,
                            "relativeDifference": scale_difference, "impliedCameraHeightM": capture_height_m},
        "floorCrossCheck": cross_check,
        "rawFrame": "+x right, +y down, +z forward (OpenCV)",
        "worldFrame": "meters; +x right, +y forward, +z up; floor below capture camera at origin",
        "matrixConvention": "row-major 4x4, multiplied by raw column vectors [x,y,z,1]",
        "rawToWorld": matrix.tolist(), "rotationRawToWorld": rotation.tolist(),
        "floorNormalRaw": chosen["upNormalRaw"], "floorOffsetRaw": chosen["cameraHeightRaw"],
        "fittedGroundTranslationM": capture_height_m,
        "floorFitMedianResidualM": scale * chosen["medianResidualRaw"],
        "floorFitP95ResidualM": scale * chosen["p95ResidualRaw"],
        "validation": {"status": "fail" if errors else "warn" if warnings else "pass",
                       "errors": errors, "warnings": warnings, "floorSpanRadiusFraction": span_fraction},
        "fits": fits, "spz": header,
        "opaqueSplatPercentileBoundsM": {"p01": np.percentile(transformed, 1, axis=0).tolist(),
                                          "p99": np.percentile(transformed, 99, axis=0).tolist()},
        "sanity": {"rotationDeterminant": float(np.linalg.det(rotation)),
                   "captureCameraWorldM": matrix[:3, 3].tolist(),
                   "floorBelowCameraWorldM": (np.einsum("ij,j->i", matrix[:3, :3], -chosen["cameraHeightRaw"] * up) + matrix[:3, 3]).tolist(),
                   "oneRawUnitLengthM": scale},
    }
    api_offset = semantics.get("ground_plane_offset")
    if isinstance(api_offset, (int, float)) and math.isfinite(api_offset):
        report["apiVsFittedGroundOffsetDifferenceM"] = float(api_offset) - capture_height_m
    if api_scale is not None and scale_difference > 0.15:
        alternative = matrix.copy()
        alternative[:3, :3] = assumed_scale * rotation
        alternative[2, 3] = camera_height_m
        report["cameraHeightAlternative"] = {"rawToWorld": alternative.tolist(), "metersPerRawUnit": assumed_scale,
                                             "cameraHeightM": camera_height_m, "referenceKnown": camera_height_known,
                                             "applied": False}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "public/worlds/factory")
    parser.add_argument("--camera-height-m", type=float, default=1.6,
                        help="Capture-height reference; an assumption unless --camera-height-known is given")
    parser.add_argument("--camera-height-known", action="store_true",
                        help="Declare that capture height was actually measured (enables stronger disagreement checks)")
    args = parser.parse_args()
    if not math.isfinite(args.camera_height_m) or not 0.3 <= args.camera_height_m <= 3:
        parser.error("camera height must be between 0.3 and 3 meters")
    report = calibrate(args.directory, args.camera_height_m, args.camera_height_known)
    (args.directory / "calibration-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["validation"]["errors"]:
        raise SystemExit("Calibration rejected; manifest unchanged. See calibration-report.json: "
                         + "; ".join(report["validation"]["errors"]))
    manifest_file = args.directory / "world.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        fields = ["estimatedMetersPerRawUnit", "metersPerRawUnit", "scaleSource", "isMetricScaleEstimated",
                  "metricScaleIsMeasuredGroundTruth", "assumedCameraHeightM", "captureCameraHeightM",
                  "cameraHeightReferenceKnown", "floorNormalRaw", "floorOffsetRaw", "rawToWorld", "validation"]
        manifest.setdefault("calibration", {}).update({key: report[key] for key in fields})
        manifest["calibration"]["report"] = "calibration-report.json"
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["scaleSource", "estimatedMetersPerRawUnit", "captureCameraHeightM",
                      "floorNormalRaw", "floorOffsetRaw", "floorFitMedianResidualM", "validation", "rawToWorld", "sanity"]}, indent=2))


if __name__ == "__main__":
    main()
