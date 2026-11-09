#!/usr/bin/env python3
"""Project, verify and assemble overlapping perspective faces of one panorama.

This tool performs deterministic camera projection and media assembly only. It
does not generate, upscale or invent scene detail; the detail faces must be
provided separately. Requires Python 3, NumPy and Pillow. All axes are raw OpenCV:
+x right, +y down, +z forward. The input/output panorama is 2:1 equirectangular.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / 'control-ui/public/worlds/factory-conveyor/panorama.jpg'
DEFAULT_FACES = ROOT / '.marble/reference/conveyor-hq/faces'
DEFAULT_OUTPUT = ROOT / 'control-ui/public/worlds/factory-conveyor-hq/source-reference.png'
# Forward, right and down camera bases; no hidden rotations or face flips.
BASES = {
    'front': ([0, 0, 1], [1, 0, 0], [0, 1, 0]),
    'right': ([1, 0, 0], [0, 0, -1], [0, 1, 0]),
    'back': ([0, 0, -1], [-1, 0, 0], [0, 1, 0]),
    'left': ([-1, 0, 0], [0, 0, 1], [0, 1, 0]),
    'up': ([0, -1, 0], [1, 0, 0], [0, 0, 1]),
    'down': ([0, 1, 0], [1, 0, 0], [0, 0, -1]),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.float32)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def panorama_directions(width: int, height: int, row_start: int, row_end: int) -> np.ndarray:
    theta = ((np.arange(width, dtype=np.float64) + .5) / width - .5) * (2 * np.pi)
    phi = ((np.arange(row_start, row_end, dtype=np.float64) + .5) / height - .5) * np.pi
    cp = np.cos(phi)[:, None]
    # Camera down (+y) is the panorama's increasing vertical coordinate.
    return np.stack(np.broadcast_arrays(np.sin(theta)[None, :] * cp,
        np.sin(phi)[:, None], np.cos(theta)[None, :] * cp), axis=-1)


def bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray, *, wrap: bool = False) -> np.ndarray:
    height, width = image.shape[:2]
    u = np.mod(u, width) if wrap else np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    x0, y0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    x1 = (x0 + 1) % width if wrap else np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx, wy = (u - x0)[..., None], (v - y0)[..., None]
    return (image[y0, x0] * (1 - wx) + image[y0, x1] * wx) * (1 - wy) + (
        image[y1, x0] * (1 - wx) + image[y1, x1] * wx) * wy


def project_face(source: np.ndarray, name: str, size: int, fov: float) -> np.ndarray:
    forward, right, down = (np.array(v, dtype=np.float64) for v in BASES[name])
    half_tan = math.tan(math.radians(fov / 2))
    grid = ((np.arange(size, dtype=np.float64) + .5) / size * 2 - 1) * half_tan
    result = np.empty((size, size, 3), dtype=np.uint8)
    for start in range(0, size, 96):
        end = min(size, start + 96)
        rays = (forward + grid[None, :, None] * right + grid[start:end, None, None] * down)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        u = (np.arctan2(rays[..., 0], rays[..., 2]) / (2 * np.pi) + .5) * source.shape[1] - .5
        v = (np.arcsin(np.clip(rays[..., 1], -1, 1)) / np.pi + .5) * source.shape[0] - .5
        result[start:end] = np.clip(np.rint(bilinear(source, u, v, wrap=True)), 0, 255).astype(np.uint8)
    return result


def face_sample(image: np.ndarray, name: str, directions: np.ndarray, fov: float):
    forward, right, down = (np.array(v, dtype=np.float64) for v in BASES[name])
    z = np.einsum('...i,i->...', directions, forward)
    safe_z = np.maximum(z, 1e-12)
    x = np.einsum('...i,i->...', directions, right) / safe_z
    y = np.einsum('...i,i->...', directions, down) / safe_z
    half_tan = math.tan(math.radians(fov / 2))
    edge = np.maximum(np.abs(x), np.abs(y))
    valid = (z > 0) & (edge <= half_tan)
    # Full weight through the central 90 degrees, smoothly taper to zero at FOV edge.
    fade = np.clip((edge - 1) / (half_tan - 1), 0, 1)
    weight = np.where(valid, .5 * (1 + np.cos(np.pi * fade)), 0)
    u = (x / half_tan + 1) * .5 * image.shape[1] - .5
    v = (y / half_tan + 1) * .5 * image.shape[0] - .5
    return bilinear(image, u, v), weight


def assemble(faces: dict[str, np.ndarray], width: int, height: int, fov: float) -> np.ndarray:
    output = np.empty((height, width, 3), dtype=np.uint8)
    for start in range(0, height, 96):
        end = min(height, start + 96)
        rays = panorama_directions(width, height, start, end)
        total = np.zeros((end - start, width, 3), dtype=np.float64)
        weights = np.zeros((end - start, width), dtype=np.float64)
        for name, face in faces.items():
            color, weight = face_sample(face, name, rays, fov)
            total += color * weight[..., None]
            weights += weight
        if np.any(weights < 1e-10):
            raise ValueError('Face projection left uncovered panorama pixels')
        output[start:end] = np.clip(np.rint(total / weights[..., None]), 0, 255).astype(np.uint8)
    return output


def load_faces(directory: Path, kind: str) -> tuple[dict[str, np.ndarray], dict]:
    faces, info = {}, {}
    for name in BASES:
        path = directory / f'{name}-{kind}.png'
        if not path.exists():
            raise ValueError(f'Missing {kind} face: {path}')
        face = read_rgb(path)
        if face.shape[0] != face.shape[1]:
            raise ValueError(f'{path.name} must be square; found {face.shape[1]}x{face.shape[0]}')
        faces[name] = face
        info[name] = {'file': path.name, 'width': face.shape[1], 'height': face.shape[0],
                      'sha256': sha256(path)}
    return faces, info


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    a -= a.mean(); b -= b.mean()
    # Elementwise reductions avoid launching a BLAS thread pool for each small crop.
    denominator = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    return float(np.sum(a * b) / denominator) if denominator > 1e-8 else None


def structure_check(inputs: dict[str, np.ndarray], details: dict[str, np.ndarray]) -> dict:
    records, errors, warnings = {}, [], []
    for name in BASES:
        def small(image):
            return np.asarray(Image.fromarray(image.astype(np.uint8)).convert('L').resize(
                (128, 128), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(1.5)), dtype=np.float64)
        base, detail = small(inputs[name]), small(details[name])
        score = correlation(base, detail)
        best, offset = -1., (0, 0)
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                y0, y1 = max(0, dy), min(128, 128 + dy)
                x0, x1 = max(0, dx), min(128, 128 + dx)
                c = correlation(base[y0:y1, x0:x1], detail[y0-dy:y1-dy, x0-dx:x1-dx])
                if c is not None and c > best:
                    best, offset = c, (dx, dy)
        informative = float(base.std()) >= 8
        if informative and score is not None and score < .65:
            errors.append(f'{name}: gross structural mismatch (low-frequency correlation {score:.3f})')
        elif informative and score is not None and score < .85:
            warnings.append(f'{name}: inspect geometry; low-frequency correlation {score:.3f}')
        if informative and max(map(abs, offset)) >= 5 and best > (score or 0) + .04:
            errors.append(f'{name}: apparent camera/framing shift {offset} at128px')
        target = Image.fromarray(inputs[name].astype(np.uint8)).resize(
            (details[name].shape[1], details[name].shape[0]), Image.Resampling.LANCZOS)
        difference = float(np.mean(np.abs(np.asarray(target, dtype=np.float64) - details[name])))
        if difference < .1:
            errors.append(f'{name}: detail face is effectively the original resample, not a changed input')
        records[name] = {'lowFrequencyCorrelation': score, 'inputLowFrequencyStd8bit': float(base.std()),
            'bestShiftAt128px': list(offset), 'bestShiftCorrelation': best,
            'meanAbsoluteDifferenceFromInputResample8bit': difference}
    return {'faces': records, 'errors': errors, 'warnings': warnings}


def overlap_check(inputs, details, fov: float) -> dict:
    rays = panorama_directions(640, 320, 0, 320)
    originals, enhanced, weights = {}, {}, {}
    for name in BASES:
        originals[name], weight = face_sample(inputs[name], name, rays, fov)
        enhanced[name], _ = face_sample(details[name], name, rays, fov)
        weights[name] = weight
    records, errors, warnings = {}, [], []
    names = list(BASES)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            if np.dot(BASES[first][0], BASES[second][0]) != 0:
                continue
            valid = (weights[first] > .05) & (weights[second] > .05)
            count = int(valid.sum())
            if count < 20:
                errors.append(f'{first}/{second}: insufficient shared projection coverage')
                continue
            base_error = np.abs(originals[first][valid] - originals[second][valid])
            detail_error = np.abs(enhanced[first][valid] - enhanced[second][valid])
            mae = float(detail_error.mean())
            if mae > 55:
                errors.append(f'{first}/{second}: gross overlap mismatch (MAE {mae:.1f}/255)')
            elif mae > 25:
                warnings.append(f'{first}/{second}: visible overlap may need review (MAE {mae:.1f}/255)')
            records[f'{first}/{second}'] = {'sampledDirections': count,
                'inputMeanAbsoluteDifference8bit': float(base_error.mean()),
                'detailMeanAbsoluteDifference8bit': mae,
                'detailP95AbsoluteDifference8bit': float(np.percentile(detail_error, 95))}
    return {'pairs': records, 'errors': errors, 'warnings': warnings}


def source_info(path: Path) -> dict:
    with Image.open(path) as im:
        width, height = im.size
    if width != 2 * height:
        raise ValueError('Source panorama must be exactly2:1')
    return {'file': str(path), 'width': width, 'height': height, 'sha256': sha256(path)}


def verify_roundtrip(source: np.ndarray, inputs, width, height, fov) -> dict:
    reconstructed = assemble(inputs, width, height, fov)
    reference = np.asarray(Image.fromarray(source.astype(np.uint8)).resize(
        (width, height), Image.Resampling.BILINEAR), dtype=np.float64)
    absolute = np.abs(reconstructed.astype(np.float64) - reference)
    # Edge/pole checks catch axis flips that could be diluted in a mostly plain scene.
    rows = max(1, height // 20)
    return {'width': width, 'height': height, 'meanAbsoluteError8bit': float(absolute.mean()),
        'p95AbsoluteError8bit': float(np.percentile(absolute, 95)),
        'topBandMeanAbsoluteError8bit': float(absolute[:rows].mean()),
        'bottomBandMeanAbsoluteError8bit': float(absolute[-rows:].mean()),
        'seamBandMeanAbsoluteError8bit': float(np.concatenate([absolute[:, :4], absolute[:, -4:]], axis=1).mean()),
        'referenceResampling': 'Pillow bilinear to output grid',
        'pass': bool(absolute.mean() <= 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['project', 'assemble', 'verify'])
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--face-dir', type=Path, default=DEFAULT_FACES)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--face-size', type=int, default=1536)
    parser.add_argument('--fov', type=float, default=100.)
    parser.add_argument('--width', type=int, default=4096)
    parser.add_argument('--height', type=int, default=2048)
    parser.add_argument('--min-face-size', type=int, default=1224)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not (90 < args.fov <= 120) or min(args.width, args.height, args.face_size) <= 0 or args.width != 2 * args.height:
        parser.error('Require90<FOV<=120, positive sizes, and output width=2*height')
    source = read_rgb(args.input)
    info = source_info(args.input)
    if args.mode == 'project':
        targets = [args.face_dir / f'{name}-input.png' for name in BASES]
        if not args.overwrite and any(p.exists() for p in targets):
            raise ValueError('Input faces already exist; use--overwrite explicitly to replace them')
        args.face_dir.mkdir(parents=True, exist_ok=True)
        for name in BASES:
            Image.fromarray(project_face(source, name, args.face_size, args.fov)).save(args.face_dir / f'{name}-input.png')
        metadata = {'version': 1, 'method': 'deterministic_perspective_projection', 'source': info,
            'faceSize': args.face_size, 'horizontalVerticalFovDegrees': args.fov,
            'overlapBeyondCentral90Degrees': args.fov - 90, 'coordinateFrame': 'raw OpenCV x-right/y-down/z-forward',
            'basesForwardRightDown': BASES}
        write_json(args.face_dir / 'projection.json', metadata)
        print(json.dumps(metadata, indent=2)); return
    inputs, input_info = load_faces(args.face_dir, 'input')
    roundtrip = verify_roundtrip(source, inputs, args.width, args.height, args.fov)
    if args.mode == 'verify':
        result = {'source': info, 'faces': input_info, 'projectionFovDegrees': args.fov,
                  'inputOnlyRoundTrip': roundtrip, 'validation': {'status': 'pass' if roundtrip['pass'] else 'fail'}}
        if args.report:
            write_json(args.report, result)
        print(json.dumps(result, indent=2))
        if not roundtrip['pass']:
            raise SystemExit('Input-only round trip failed (mean absolute error exceeds2/255)')
        return
    details, detail_info = load_faces(args.face_dir, 'detail')
    errors, warnings = [], []
    if not roundtrip['pass']:
        errors.append('Input-only round trip failed: check camera axes/FOV/face projections before assembly')
    for name, item in detail_info.items():
        if min(item['width'], item['height']) < args.min_face_size:
            errors.append(f'{name}: actual detail face below {args.min_face_size}px minimum')
        item['central90DegreeLinearWidthPixels'] = item['width'] / math.tan(math.radians(args.fov / 2))
        item['actualInputWidthPixels'] = input_info[name]['width']
    structure = structure_check(inputs, details)
    overlaps = overlap_check(inputs, details, args.fov)
    errors += structure['errors'] + overlaps['errors']
    warnings += structure['warnings'] + overlaps['warnings']
    report = {'version': 1, 'method': 'perspective_face_reprojection_and_cosine_overlap_blending',
        'detailProvenance': 'Caller-supplied enhanced faces; image generation provenance is recorded separately.',
        'source': info, 'inputFaces': input_info, 'detailFaces': detail_info,
        'projectionFovDegrees': args.fov, 'centralFaceFovDegrees': 90,
        'overlapBeyondCentral90Degrees': args.fov - 90, 'inputOnlyRoundTrip': roundtrip,
        'structureChecks': structure, 'overlapChecks': overlaps,
        'output': {'file': str(args.output), 'width': args.width, 'height': args.height},
        'coordinateFrame': 'raw OpenCV x-right/y-down/z-forward', 'basesForwardRightDown': BASES,
        'samplingNote': '4K describes assembled raster dimensions. Cube-to-equirectangular angular sampling varies; no native single-image4K or measured new-detail guarantee is inferred from resizing alone.',
        'validation': {'status': 'fail' if errors else 'warn' if warnings else 'pass', 'errors': errors, 'warnings': warnings}}
    report_file = args.report or args.output.with_name('source-reference-report.json')
    if errors:
        rejected = report_file.with_name(report_file.stem + '-rejected.json')
        write_json(rejected, report)
        raise SystemExit('HQ panorama rejected before output generation: ' + '; '.join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(assemble(details, args.width, args.height, args.fov)).save(args.output)
    report['output']['sha256'] = sha256(args.output)
    report['output']['bytes'] = args.output.stat().st_size
    write_json(report_file, report)
    print(json.dumps({'output': report['output'], 'validation': report['validation'],
                      'roundTripMeanAbsoluteError8bit': roundtrip['meanAbsoluteError8bit']}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from None
