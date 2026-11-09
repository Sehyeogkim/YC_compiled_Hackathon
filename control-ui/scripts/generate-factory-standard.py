#!/usr/bin/env python3
"""Generate one Marble 1.1 factory from a reference panorama; resume without repaying."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.marble'
OUTPUT = ROOT / 'control-ui/public/worlds/factory-standard'
SOURCE = CACHE / 'reference/university-workshop.jpg'
API = 'https://api.worldlabs.ai/marble/v1/'
MODEL = 'marble-1.1'
PROMPT = ('A real spacious industrial manufacturing workshop with concrete floor, steel roof beams, '
          'skylights, machinery, benches and storage around the perimeter. Preserve the supplied '
          '360 degree photograph, its architecture, floor and realistic human scale. Keep the open '
          'central floor unobstructed. No additional robot, conveyor, people or foreground machinery. '
          'Sharp, coherent walls, doors and ceiling; natural industrial daylight.')


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def main():
    global OUTPUT, SOURCE, PROMPT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['workshop', 'conveyor', 'conveyor-hq'], default='conveyor')
    parser.add_argument('--generate', action='store_true', help='Authorize one 1500-credit panorama generation')
    args = parser.parse_args()
    conveyor = args.profile.startswith('conveyor')
    high_detail = args.profile == 'conveyor-hq'
    job = 'factory-conveyor-hq' if high_detail else 'factory-conveyor' if conveyor else 'factory-standard'
    OUTPUT = ROOT / 'control-ui/public/worlds' / job
    if conveyor:
        SOURCE = OUTPUT / 'source-reference.png' if high_detail else CACHE / 'reference/conveyor-reference-master.png'
        if not SOURCE.exists() and (OUTPUT / 'source-reference.png').exists():
            SOURCE = OUTPUT / 'source-reference.png'
        PROMPT = ('A clean modern aluminum beverage can recycling and conveyor sorting factory. '
            'Preserve the exact supplied panorama layout: narrow black rubber conveyors on galvanized '
            'steel frames, blue optical sorting machines, silver chutes and hoppers, cans and compacted '
            'aluminum bales around the perimeter. Flat gray concrete floor with yellow safety lines, '
            'gray metal wall panels, regular roof trusses, bright neutral industrial lighting. '
            'Preserve the empty central robot service bay and keep it unobstructed for a simulated '
            'robot and 1.4 meter conveyor to be inserted later. Background conveyors are human-scale '
            'about 0.45 m wide and 0.87 m high, personnel doors 2.1 m tall. '
            'No central robot, central workbench, central conveyor, people or woodworking equipment. '
            'Sharp coherent photographic industrial architecture and realistic human scale.')
        if high_detail:
            PROMPT += (' Preserve the high-detail reference image faithfully: sharp galvanized frame joints, '
                'individual cans, panel seams, bolts, roof ribs and concrete grain. The stacked compacted '
                'metal bales must remain NEUTRAL SILVER GRAY without colorful printed can labels; '
                'preserve their exact shape, stack and wooden pallets. Do not reintroduce multicolored bales.')
    CACHE.mkdir(exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ['.env', 'control-ui/.env.local']:
        path = ROOT / name
        if path.exists():
            for line in path.read_text().splitlines():
                if '=' in line and not line.lstrip().startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('\"\''))
    key = os.environ.get('WORLDLABS_API_KEY') or os.environ.get('WLT_API_KEY')
    if not key:
        raise RuntimeError('Set WORLDLABS_API_KEY in the repository .env')

    def api(endpoint, body=None):
        req = urllib.request.Request(API + endpoint,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'WLT-Api-Key': key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'World Labs {endpoint}: HTTP {error.code}: {error.read().decode()[:400]}') from None

    operation_path = CACHE / f'{job}-operation.json'
    request_marker = CACHE / f'{job}-request.json'
    if operation_path.exists():
        operation = json.loads(operation_path.read_text())
        print('Resuming existing standard generation; no new paid request.', flush=True)
    else:
        if not args.generate:
            raise RuntimeError('Use --generate for one new 1500-credit Marble 1.1 generation.')
        if request_marker.exists():
            raise RuntimeError('A paid request was previously attempted; inspect account operations before retrying.')
        balance = api('credits')['remaining_credits']
        if balance < 1500:
            raise RuntimeError(f'Insufficient balance: {balance}; need 1500 credits.')
        if not SOURCE.exists():
            if conveyor:
                raise RuntimeError('Prepare .marble/reference/conveyor-reference-master.png first.')
            SOURCE.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request('https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/university_workshop.jpg',
                headers={'User-Agent': 'WeRescue factory background builder', 'Referer': 'https://polyhaven.com/'})
            with urllib.request.urlopen(req, timeout=180) as response:
                SOURCE.write_bytes(response.read())
        from PIL import Image
        with Image.open(SOURCE) as reference:
            if reference.width != 2 * reference.height:
                raise RuntimeError('Input must be a true 2:1 equirectangular panorama.')
            if high_detail and reference.width < 4096:
                raise RuntimeError('HQ regeneration requires a verified input at least4096 pixels wide.')
        media_file = CACHE / f'{job}-media.json'
        if media_file.exists():
            media = json.loads(media_file.read_text())
        else:
            media = api('media-assets:prepare_upload', {'file_name': SOURCE.name, 'kind': 'image', 'extension': SOURCE.suffix.lstrip('.')})
            upload = media['upload_info']
            req = urllib.request.Request(upload['upload_url'], data=SOURCE.read_bytes(),
                method=upload['upload_method'], headers=upload.get('required_headers', {}))
            with urllib.request.urlopen(req, timeout=180) as response:
                response.read()
            save(media_file, media)
        asset = media['media_asset']
        body = {'display_name': 'WeRescue can recycling conveyor factory HQ' if high_detail else 'WeRescue can recycling conveyor factory' if conveyor else 'WeRescue real industrial workshop', 'model': MODEL,
            'permission': {'public': False},
            'world_prompt': {'type': 'image', 'is_pano': True, 'disable_recaption': True,
                'image_prompt': {'source': 'media_asset', 'media_asset_id': asset.get('id') or asset.get('media_asset_id')},
                'text_prompt': PROMPT}}
        save(request_marker, {'request': body, 'balanceBefore': balance, 'sourceSha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest()})
        print(f'Balance {balance}; starting ONE Marble 1.1 panorama generation (1500 credits).', flush=True)
        operation = api('worlds:generate', body)
        save(operation_path, operation)
    for _ in range(180):
        if operation.get('done'):
            break
        time.sleep(5)
        operation = api('operations/' + operation['operation_id'])
        save(operation_path, operation)
        print(operation.get('metadata', {}).get('progress', {}).get('description', 'Generating…'), flush=True)
    if not operation.get('done') or operation.get('error'):
        raise RuntimeError(operation.get('error') or 'Still pending; rerun to resume without paying again.')
    response = operation.get('response') or {}
    world_id = response.get('world_id') or response.get('id') or operation['metadata']['world_id']
    record = api('worlds/' + world_id)
    world = record.get('world', record)
    save(CACHE / f'{job}-world.json', world)
    assets = world['assets']
    urls = {
        'factory-100k.spz': assets['splats']['spz_urls']['100k'],
        'factory-500k.spz': assets['splats']['spz_urls']['500k'],
        'factory-full.spz': assets['splats']['spz_urls']['full_res'],
        'collider.glb': assets.get('mesh', {}).get('collider_mesh_url'),
        'panorama.jpg': assets.get('imagery', {}).get('pano_url'),
        'thumbnail.jpg': assets.get('thumbnail_url'),
    }

    def download(item):
        name, url = item
        if not url:
            return None
        dest = OUTPUT / name
        if not dest.exists():
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=180) as response:
                        data = response.read()
                    temp = dest.with_suffix(dest.suffix + '.tmp')
                    temp.write_bytes(data)
                    temp.replace(dest)
                    break
                except (OSError, urllib.error.URLError):
                    if attempt == 2:
                        raise
                    time.sleep(2)
        print(f'Downloaded {name}: {dest.stat().st_size} bytes', flush=True)
        return name, dest.stat().st_size

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = dict(item for item in pool.map(download, urls.items()) if item)
    from PIL import Image
    with Image.open(SOURCE) as image:
        source_width, source_height = image.size
        image.resize((1024, 512), Image.Resampling.LANCZOS).save(OUTPUT / 'lighting-pano.jpg', quality=92)
        image.thumbnail((4096, 2048), Image.Resampling.LANCZOS)
        image.convert('RGB').save(OUTPUT / 'source-reference.jpg', quality=95)
    for name in ['lighting-pano.jpg', 'source-reference.jpg']:
        files[name] = (OUTPUT / name).stat().st_size
    manifest = {
        'version': 2, 'worldId': world_id, 'model': MODEL, 'source': world.get('world_marble_url'),
        'generatedAt': world.get('created_at'), 'prompt': PROMPT,
        'input': {'type': 'ai_regenerated_cube_faces_reprojected_to_panorama' if high_detail else 'generated_equirectangular_reference' if conveyor else 'real_equirectangular_panorama', 'width': source_width, 'height': source_height,
            **({'generator': 'OpenAI built-in image_gen + cubemap reprojection' if high_detail else 'OpenAI built-in image_gen', 'theme': 'aluminum can recycling conveyor sorting factory',
                **({'sourceReport': 'source-reference-report.json', 'prompts': 'reference-prompts.json', 'baleMaterial': 'neutral silver-gray compressed aluminum'} if high_detail else {})} if conveyor else {
            'page': 'https://polyhaven.com/a/university_workshop', 'author': 'Savva Zakharov',
            'license': 'CC0-1.0', 'licenseUrl': 'https://polyhaven.com/license'}),
            'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'captureHeightM': None},
        'splats': {'100k': 'factory-100k.spz', '500k': 'factory-500k.spz', 'full': 'factory-full.spz'},
        'semantics': assets['splats'].get('semantics_metadata'),
        'lighting': 'lighting-pano.jpg',
        'calibration': {'scaleCorrection': 1, 'groundCorrectionM': 0, 'yawDegrees': 0, 'translationM': [0, 0, 0]},
        'files': files, 'generationCost': operation.get('cost'),
    }
    manifest_path = OUTPUT / 'world.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get('worldId') == world_id:
            manifest['calibration'] = previous['calibration']
    save(manifest_path, manifest)
    subprocess.run([sys.executable, str(Path(__file__).with_name('calibrate-factory-world.py')),
        '--directory', str(OUTPUT)], check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name('calibrate-visible-floor.py')),
        '--directory', str(OUTPUT)], check=True)
    print(json.dumps({'worldId': world_id, 'semantics': manifest['semantics'],
        'cost': operation.get('cost'), 'remainingCredits': api('credits')['remaining_credits']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
