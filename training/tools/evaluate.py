#!/usr/bin/env python3
"""Summarize paired measured P vs P+R trials and draw an SVG from real outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
import math
from pathlib import Path


class EvaluationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0, center - half), min(1, center + half)]


def summarize(data: dict) -> dict:
    require(isinstance(data, dict), 'evaluation must be a JSON object')
    require(data.get('schema_version') == 'werescue.task_eval.v1', 'unsupported evaluation schema_version')
    protocol = data.get('protocol_id')
    require(isinstance(protocol, str) and bool(protocol), 'protocol_id required')
    trials = data.get('trials')
    require(isinstance(trials, list) and bool(trials), 'measured trials required')
    arms: dict[str, dict[tuple, dict]] = {'P': {}, 'P+R': {}}
    checkpoints: dict[str, set[str]] = {'P': set(), 'P+R': set()}
    for row in trials:
        require(isinstance(row, dict), 'trial must be an object')
        arm = row.get('arm')
        require(arm in arms, 'arm must be P or P+R')
        require(row.get('split') == 'test', 'only held-out test trials can enter comparison')
        require(row.get('control_source') == 'policy', 'comparison requires policy execution')
        require(row.get('contains_human_intervention') is False, 'autonomous task comparison cannot include human intervention')
        require(type(row.get('success')) is bool, 'success must be a measured boolean')
        require(row.get('success_predicate_verified') is True, 'native success predicate verification required')
        require(type(row.get('seed')) is int, 'integer seed required')
        for field in ('task_id', 'start_state_id', 'checkpoint', 'environment_id', 'trace_ref'):
            require(isinstance(row.get(field), str) and bool(row[field]), f'{field} required')
        key = (row['environment_id'], row['task_id'], row['start_state_id'], row['seed'])
        require(key not in arms[arm], f'duplicate trial for {arm}: {key}')
        arms[arm][key] = row
        checkpoints[arm].add(row['checkpoint'])
    require(bool(arms['P']) and arms['P'].keys() == arms['P+R'].keys(),
            'P and P+R must have exactly the same held-out starts and seeds')
    require(all(len(values) == 1 for values in checkpoints.values()), 'one fixed checkpoint per arm required')
    require(checkpoints['P'] != checkpoints['P+R'], 'P+R must identify a distinct trained checkpoint')
    training_ids = data.get('training_initial_state_ids', [])
    require(isinstance(training_ids, list) and all(isinstance(value, str) and bool(value.strip()) for value in training_ids),
            'training_initial_state_ids must be a list of nonempty canonical string IDs')
    forbidden = set(training_ids)
    require(all(key[2] not in forbidden for key in arms['P']), 'evaluation start overlaps training initial states')
    summary = {}
    for arm, rows in arms.items():
        successes = sum(row['success'] for row in rows.values())
        summary[arm] = {'successes': successes, 'attempts': len(rows),
                        'success_rate': successes / len(rows), 'wilson_95': wilson(successes, len(rows)),
                        'checkpoint': next(iter(checkpoints[arm]))}
    paired = Counter()
    for key, before in arms['P'].items():
        after = arms['P+R'][key]
        paired[f'{int(before["success"])}{int(after["success"])}'] += 1
    return {'schema_version': 'werescue.task_eval_summary.v1', 'protocol_id': protocol,
            'arms': summary, 'paired_outcomes': {name: paired[name] for name in ('00', '01', '10', '11')},
            'delta_percentage_points': 100 * (summary['P+R']['success_rate'] - summary['P']['success_rate']),
            'confidence_note': 'Wilson 95% intervals describe each proportion; they are not a paired-difference significance test.',
            'training_overlap_check': 'checked_supplied_initial_state_ids' if forbidden else 'no_training_manifest_supplied'}


def chart_svg(summary: dict) -> str:
    # Fixed 0–100% scale; display denominators and uncertainty, never training loss as success.
    bars = []
    for i, arm in enumerate(('P', 'P+R')):
        row = summary['arms'][arm]
        x, base, height = 170 + i * 255, 300, row['success_rate'] * 220
        low, high = row['wilson_95']
        color = '#92a1b5' if arm == 'P' else '#0d9488'
        bars.append(f'<rect x="{x}" y="{base-height:.2f}" width="130" height="{height:.2f}" rx="4" fill="{color}"/>')
        bars.append(f'<path d="M{x+65} {base-high*220:.2f}V{base-low*220:.2f}M{x+55} {base-high*220:.2f}h20M{x+55} {base-low*220:.2f}h20" stroke="#172033" stroke-width="2"/>')
        label = f'{row["successes"]}/{row["attempts"]} · {row["success_rate"]*100:.1f}%'
        bars.append(f'<text x="{x+65}" y="345" text-anchor="middle" font-size="18">{escape(arm)}</text><text x="{x+65}" y="370" text-anchor="middle" font-size="14">{label}</text>')
    grid = ''.join(f'<path d="M90 {300-i*55}H650" stroke="#e5e9ef"/><text x="72" y="{305-i*55}" text-anchor="end" font-size="12">{i*25}%</text>' for i in range(5))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="720" height="440" viewBox="0 0 720 440"><rect width="720" height="440" fill="#fff"/><g font-family="Arial,sans-serif" fill="#172033"><text x="40" y="34" font-size="21" font-weight="bold">Held-out task completion: P vs P+R</text><text x="40" y="57" font-size="12">Matched start states and seeds · measured outcomes · Wilson 95% intervals</text>{grid}{''.join(bars)}<text x="40" y="412" font-size="12">Protocol: {escape(summary['protocol_id'])} · paired trials: {summary['arms']['P']['attempts']}</text></g></svg>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--svg', type=Path)
    args = parser.parse_args()
    try:
        summary = summarize(json.loads(args.input.read_text()))
    except EvaluationError as error:
        parser.exit(2, f'evaluation rejected: {error}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    if args.svg:
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(chart_svg(summary))
    print(f'Measured paired trials: {summary["arms"]["P"]["attempts"]}; summary: {args.output}')


if __name__ == '__main__':
    main()
