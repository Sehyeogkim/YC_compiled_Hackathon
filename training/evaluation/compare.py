"""Compare completed native baseline pilots with verified paired start states."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


def paired_comparison(libero, base, protocols=None, runtimes=None):
    for summary, arm in ((libero, 'libero'), (base, 'base-adapted')):
        if summary.get('arm') != arm:
            raise ValueError(f'Expected {arm} summary')
        if summary['completed_trials'] != summary['expected_trials'] or not summary['completed_trials']:
            raise ValueError('Both evaluation runs must finish before making this comparison')
        if summary.get('phase') not in ('pilot', 'final'):
            raise ValueError('Unknown evaluation phase')
    if libero['phase'] != base['phase']:
        raise ValueError('Cannot combine development pilot and final evaluation')
    if protocols is not None:
        common = [{k: v for k, v in p.items() if k not in ('arm', 'policy', 'protocol_sha256')}
                  for p in protocols]
        if len(common) != 2 or common[0] != common[1]:
            raise ValueError('Shared execution protocols differ')
        for summary, protocol in zip((libero, base), protocols):
            if summary['protocol_sha256'] != protocol['protocol_sha256']:
                raise ValueError('Summary does not identify supplied protocol')
    if runtimes is not None:
        for field in ('packages', 'assets', 'source_hashes', 'evaluator_source_sha256', 'gpu'):
            if runtimes[0][field] != runtimes[1][field]:
                raise ValueError(f'Shared runtime differs at {field}')
    a = {row['trial_id']: row for row in libero['trials']}
    b = {row['trial_id']: row for row in base['trials']}
    if a.keys() != b.keys() or len(a) != libero['completed_trials']:
        raise ValueError('Trial keys must be identical and unique')
    outcomes = {key: 0 for key in ('00', '01', '10', '11')}
    for key, first in a.items():
        second = b[key]
        for field in ('initial_state_id', 'policy_rng_seed', 'env_seed',
                      'initial_native_state_sha256', 'initial_objects',
                      'initial_xml_sha256', 'suite', 'task_id', 'language'):
            if first[field] != second[field]:
                raise ValueError(f'Unmatched starting condition for {key}: {field}')
        outcomes[f'{int(first["success"])}{int(second["success"])}'] += 1
    positions = [row['initial_objects']['alphabet_soup_1'] for row in a.values()]
    unique_positions = {tuple(round(value, 4) for value in pos) for pos in positions}
    return {
        'schema_version': 'werescue.baseline_comparison.v1',
        'phase': libero['phase'], 'paired_trials': len(a),
        'unique_initial_state_ids': len({row['initial_state_id'] for row in a.values()}),
        'unique_can_positions_at_0_1mm_resolution': len(unique_positions),
        'can_position_span_m': [max(p[i] for p in positions) - min(p[i] for p in positions) for i in range(3)],
        'initial_states_match': True,
        'shared_execution_protocol_verified': protocols is not None,
        'shared_runtime_verified': runtimes is not None,
        'experiment_condition': protocols[0].get('condition', {'name': 'official_initial_states'}) if protocols else None,
        'arms': {arm: {key: summary[key] for key in (
            'successes', 'completed_trials', 'success_rate', 'wilson_95', 'policy')}
            for arm, summary in (('libero', libero), ('base-adapted', base))},
        'paired_outcomes_libero_then_base': outcomes,
        'claim_scope': 'Comparison of LIBERO checkpoint versus base weights with declared LIBERO I/O and normalization adaptation; not raw-base zero-shot.',
        'uncertainty_note': 'Repeated policy RNG seeds share layouts. Wilson intervals are descriptive and ignore layout clustering.',
        'raw_unadapted_base_success_rate': None,
    }


def chart_svg(result):
    title = 'Development pilot' if result['phase'] == 'pilot' else 'Final evaluation'
    condition = result.get('experiment_condition') or {}
    if condition.get('name') == 'alphabet_soup_target_position_jitter':
        title = f'Position pilot: ±{condition["jitter_max_abs_xy_m"]*100:g}cm per axis'
    grid = ''.join(f'<path d="M90 {320-i*55}H700" stroke="#e6ebf0"/><text x="72" y="{325-i*55}" text-anchor="end" font-size="12">{i*25}%</text>' for i in range(5))
    marks = []
    for arm, x, label, color in (
        ('libero', 190, 'SmolVLA-LIBERO', '#0d9488'),
        ('base-adapted', 470, 'Base + LIBERO adapter', '#64748b'),
    ):
        row = result['arms'][arm]
        y = 320 - 220 * row['success_rate']
        low, high = row['wilson_95']
        marks.append(f'<rect x="{x}" y="{y}" width="130" height="{320-y}" rx="4" fill="{color}"/>')
        marks.append(f'<path d="M{x+65} {320-high*220}V{320-low*220}M{x+55} {320-high*220}h20M{x+55} {320-low*220}h20" stroke="#172033" stroke-width="2"/>')
        marks.append(f'<text x="{x+65}" y="353" text-anchor="middle" font-size="16">{escape(label)}</text><text x="{x+65}" y="380" text-anchor="middle" font-size="16" font-weight="bold">{row["successes"]}/{row["completed_trials"]} · {row["success_rate"]*100:.0f}%</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="460" viewBox="0 0 800 460"><rect width="800" height="460" fill="white"/><g font-family="Arial,sans-serif" fill="#172033"><text x="40" y="38" font-size="24" font-weight="bold">{title}: native LIBERO task completion</text><text x="40" y="64" font-size="14">{result["unique_initial_state_ids"]} matched initial states × 2 seeds · {result["unique_can_positions_at_0_1mm_resolution"]} observed can positions</text>{grid}{"".join(marks)}<text x="40" y="420" font-size="12">Base arm uses LIBERO I/O and calibration statistics; no additional gradient updates.</text><text x="40" y="440" font-size="12">Error bars: descriptive Wilson 95% intervals; repeated starts are correlated.</text></g></svg>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libero', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = (args.libero, args.base)
    protocols = [json.loads((p.parent / 'protocol.json').read_text()) for p in paths]
    runtimes = [json.loads((p.parent / 'runtime.json').read_text()) for p in paths]
    result = paired_comparison(json.loads(args.libero.read_text()), json.loads(args.base.read_text()),
                               protocols, runtimes)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'comparison.svg').write_text(chart_svg(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
