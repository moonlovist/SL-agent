"""Run a labelled pilot without giving reference labels to the reasoning agent."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from expert_agent.agents.llm_assessment import PROMPT_VERSION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalogue', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--judgment', choices=('rules', 'deepseek'), default='rules')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    with args.catalogue.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    images = {p.stem: p.resolve() for p in args.images.rglob('*.fits') if p.stem == p.parent.name}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    def run(row):
        object_id = row['id_str']
        source = images.get(object_id)
        if source is None:
            return {'id_str': object_id, 'exit_code': 1, 'status': 'missing_input'}
        folder = output / object_id
        summary_path = folder / 'summary.json'
        if args.resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            review = summary.get('llm_assessment') or {}
            if (args.judgment == 'rules' or (review.get('status') == 'completed' and review.get('prompt_version') == PROMPT_VERSION)) and (summary.get('decision_audit') or {}).get('audit_passed'):
                return {'id_str': object_id, 'exit_code': 0, 'status': 'already_completed'}
        observation = folder / 'observation.json'
        if observation.exists():
            command = [sys.executable, '-m', 'expert_agent', 'reason', str(observation),
                       '--output', str(folder), '--judgment', args.judgment]
        else:
            command = [sys.executable, '-m', 'expert_agent', 'analyze', str(source),
                       '--output', str(output), '--judgment', args.judgment]
        task_env = dict(os.environ, PYTHONUNBUFFERED='1')
        with (output / f'{object_id}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=task_env)
        review = {}
        if summary_path.exists():
            review = json.loads(summary_path.read_text(encoding='utf-8')).get('llm_assessment') or {}
        entry = {'id_str': object_id, 'exit_code': result.returncode,
                 'llm_status': review.get('status'), 'llm_error': review.get('error')}
        print(json.dumps(entry), flush=True)
        return entry
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        completed = list(pool.map(run, rows))
    (output / 'pilot_run_status.json').write_text(json.dumps(completed, indent=2), encoding='utf-8')
    return 0 if all(row['exit_code'] == 0 for row in completed) else 1


if __name__ == '__main__':
    raise SystemExit(main())
