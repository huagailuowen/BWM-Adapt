#!/usr/bin/env python3
"""Build the selected formal ablation table from recorded scores on a compute node."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
METRICS = [
    ('centroid_ade_px', 'Object ADE (px)', 2, False, 1),
    ('centroid_fde_px', 'Object FDE (px)', 2, False, 1),
    ('psnr_multiview', 'PSNR (MV)', 3, True, 1),
    ('ssim_multiview', 'SSIM (MV)', 4, True, 1),
    ('lpips_multiview', 'LPIPS (MV)', 4, False, 1),
    ('action_success_all', 'Action success', 0, True, 100),
]


def font(size, bold=False):
    names = (
        ['/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
         '/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf']
        if bold else
        ['/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
         '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf']
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    raise FileNotFoundError('A serif font is required for the paper table.')


def formatted(row, metric):
    key, _, precision, _, scale = metric
    value = row.get(key)
    if value is None:
        return ''
    suffix = '%' if scale == 100 else ''
    return f'{value * scale:.{precision}f}{suffix}'


def render(title, rows, stem):
    widths = [950, 340, 340, 300, 300, 300, 420]
    left, top, header_height, row_height = 90, 225, 95, 100
    width = sum(widths) + 2 * left
    bottom = top + header_height + len(rows) * row_height
    height = bottom + 255
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g fill="#111111" font-family="DejaVu Serif, Liberation Serif, serif">',
    ]
    fonts = {}

    def text(x, y, value, size=30, bold=False, align='middle'):
        key = (size, bold)
        if key not in fonts:
            fonts[key] = font(size, bold)
        draw.text((x, y), value, font=fonts[key], fill='#111111',
                  anchor='lm' if align == 'start' else 'mm')
        weight = 700 if bold else 400
        svg.append(f'<text x="{x}" y="{y}" text-anchor="{align}" dominant-baseline="middle" font-size="{size}" font-weight="{weight}">{html.escape(value)}</text>')

    def line(y, thickness):
        right = width - left
        draw.line((left, y, right, y), fill='#111111', width=thickness)
        svg.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#111111" stroke-width="{thickness}"/>')

    text(width / 2, 75, title, 48, True)
    text(width / 2, 145, '5 ID + 5 OOD environments; K=1 informative support; 90 disjoint queries', 30)
    line(top, 5)
    line(top + header_height, 2)
    line(bottom, 5)
    headers = ['Method'] + [label + (' (higher)' if higher else ' (lower)')
                            for _, label, _, higher, _ in METRICS]
    # Keep direction labels on their own line to avoid crowded column headers.
    x = left
    for index, header in enumerate(headers):
        center = x + widths[index] / 2
        if index == 0:
            text(x + 15, top + header_height / 2, header, 32, True, 'start')
        else:
            label, direction = header.rsplit(' (', 1)
            text(center, top + 32, label, 27, True)
            text(center, top + 65, direction.rstrip(')') + ' is better', 23)
        x += widths[index]
    best = {}
    for key, _, _, higher, _ in METRICS:
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            best[key] = (max if higher else min)(values)
    for index, row in enumerate(rows):
        y = top + header_height + row_height * (index + 0.5)
        text(left + 15, y, row['label'], 31, index == 0, 'start')
        x = left + widths[0]
        for column, metric in enumerate(METRICS, 1):
            key = metric[0]
            value = row.get(key)
            bold = value is not None and key in best and math.isclose(value, best[key], rel_tol=1e-10)
            text(x + widths[column] / 2, y, formatted(row, metric), 33, bold)
            x += widths[column]
        if index == 0:
            line(top + header_height + row_height, 1)
    notes = [
        'MV: concatenated main + wrist views. Object ADE/FDE: main-view centroid error in pixels.',
        'Action success: 25 reachable environment-target decisions. Bold: best available value per column.',
        'Blank cells denote unfinished training or unavailable formal metrics, never zero.',
        'Historical initialization runs retain their training schedules; these are not all strict one-factor controls.',
    ]
    for index, note in enumerate(notes):
        text(left, bottom + 55 + 46 * index, note, 25, align='start')
    svg.extend(['</g>', '</svg>'])
    stem.with_suffix('.svg').write_text('\n'.join(svg) + '\n')
    image.save(stem.with_suffix('.png'), dpi=(300, 300))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/evaluation/event80_formal_ablation.yaml')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this postprocessing command on a Slurm compute node, not the login node.')
    config = yaml.safe_load(args.config.read_text())
    source = ROOT / config['source_scoreboard']
    source_bytes = source.read_bytes()
    scores = {row['method']: row for row in csv.DictReader(source_bytes.decode().splitlines())}
    additional_sources = {}
    rows = []
    for spec in config['rows']:
        row = dict(spec)
        row['source_scoreboard'] = spec.get('source_scoreboard', config['source_scoreboard'])
        selected_scores = scores
        if spec.get('source_scoreboard'):
            selected_source = ROOT / spec['source_scoreboard']
            selected_scores = {}
            if selected_source.is_file():
                payload = selected_source.read_bytes()
                selected_scores = {item['method']: item for item in csv.DictReader(payload.decode().splitlines())}
                source_record = {'sha256': hashlib.sha256(payload).hexdigest()}
                source_protocol = selected_source.parent / 'protocol.json'
                if source_protocol.is_file():
                    source_record['protocol_snapshot'] = json.loads(source_protocol.read_text())
                additional_sources[spec['source_scoreboard']] = source_record
        score = selected_scores.get(spec['method']) if spec['status'] in ('scored', 'metrics_pending') else None
        if spec['status'] == 'scored' and score is None:
            raise KeyError(f"Selected scored method missing from source: {spec['method']}")
        if score is not None:
            row['status'] = 'scored'
        for key, *_ in METRICS:
            value = score.get(key) if score else None
            row[key] = float(value) if value not in (None, '') else None
        for key in ['environment_count', 'query_count', 'action_eligible_decisions_all']:
            value = score.get(key) if score else None
            row[key] = int(float(value)) if value not in (None, '') else None
        rows.append(row)
    output = ROOT / config['output_dir']
    output.mkdir(parents=True, exist_ok=True)
    fields = ['method', 'label', 'candidate', 'source_job', 'checkpoint_step', 'status', 'setting', 'source_scoreboard',
              'environment_count', 'query_count', 'action_eligible_decisions_all'] + [m[0] for m in METRICS]
    with (output / 'scoreboard.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / 'scoreboard.json').write_text(json.dumps(rows, indent=2) + '\n')
    protocol = dict(config)
    protocol['source_scoreboard_sha256'] = hashlib.sha256(source_bytes).hexdigest()
    protocol['additional_scoreboard_sources'] = additional_sources
    protocol['render_job_id'] = os.environ['SLURM_JOB_ID']
    protocol['source_protocol_snapshot'] = json.loads((ROOT / config['source_protocol']).read_text())
    protocol['metrics_recomputed'] = False
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    headers = ['Method'] + [m[1] + (' higher' if m[3] else ' lower') for m in METRICS]
    markdown = [f"# {config['title']}", '', config['selection_note'], '',
                '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        markdown.append('| ' + ' | '.join([row['label']] + [formatted(row, m) for m in METRICS]) + ' |')
    markdown.extend(['', '## Protocol and scope', '',
                     'Event80; 5 ID + 5 OOD environments; K=1 informative support; nine disjoint queries per environment.',
                     'Metrics are copied from the recorded formal scoreboard. No new rollout or metric computation is performed.', '',
                     *['- ' + note for note in config['notes']], '', '## Run provenance', '',
                     '| Method | Checkpoint step | Source job | Status | Setting |',
                     '| --- | --- | --- | --- | --- |'])
    for row in rows:
        markdown.append('| ' + ' | '.join(str(row.get(key, '')) for key in
                         ['label', 'checkpoint_step', 'source_job', 'status', 'setting']) + ' |')
    markdown.extend(['', f"Source scores: `{config['source_scoreboard']}`.",
                     f"Support/query identities: `{config['support_query_manifest']}`.",
                     'Machine-readable selected runs and the source protocol snapshot are in `protocol.json`.', ''])
    (output / 'formal_ablation.md').write_text('\n'.join(markdown))
    render(config['title'], rows, output / 'event80_formal_ablation_table')
    print(f'Wrote formal ablation table: {output}', flush=True)
    print(f'Scored rows: {sum(row["status"] == "scored" for row in rows)}; pending rows: {sum(row["status"] != "scored" for row in rows)}', flush=True)


if __name__ == '__main__':
    main()
