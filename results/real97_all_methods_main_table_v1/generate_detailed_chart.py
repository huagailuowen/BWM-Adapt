#!/usr/bin/env python3
"""Render a simulation-style detailed real-task table from the existing main CSV."""
import csv
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2220, 2050


class Chart:
    def __init__(self):
        self.image = Image.new('RGB', (WIDTH, HEIGHT), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = {}
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
                    '<rect width="100%" height="100%" fill="white"/>',
                    '<desc>Detailed real-world metrics from the current published main CSV. ADE and FDE are separate columns. Ours means Stage2, not Stage1. Stick action uses a post-hoc selected 36-query subset; other Stick metrics use full-test queries with a common tracking mask. Soft uses six shared environments; its onset Success@6 degrees uses eleven GT-onset-positive queries and a post-hoc-selected tolerance.</desc>',
                    '<g fill="#111" font-family="DejaVu Serif,serif">']

    def text(self, x, y, value, size=24, bold=False, align='left', color='#111'):
        key = (size, bold)
        if key not in self.fonts:
            filename = 'DejaVuSerif-Bold.ttf' if bold else 'DejaVuSerif.ttf'
            self.fonts[key] = ImageFont.truetype(filename, size)
        self.draw.text((x, y), value, font=self.fonts[key], fill=color,
                       anchor={'left': 'lm', 'center': 'mm', 'right': 'rm'}[align])
        self.svg.append(f'<text x="{x}" y="{y}" text-anchor="{dict(left="start", center="middle", right="end")[align]}" dominant-baseline="middle" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(value)}</text>')

    def line(self, y, width=2):
        self.draw.line((60, y, WIDTH - 60, y), fill='#111', width=width)
        self.svg.append(f'<line x1="60" x2="{WIDTH - 60}" y1="{y}" y2="{y}" stroke="#111" stroke-width="{width}"/>')

    def save(self, stem):
        (OUT / (stem + '.svg')).write_text('\n'.join(self.svg + ['</g>', '</svg>']) + '\n')
        self.image.save(OUT / (stem + '.png'), dpi=(300, 300))


def main():
    # Some historical CSV rows contain literal escaped line endings.
    source = (OUT / 'real97_all_methods_main_table.csv').read_text().replace('\\r\\n', '\n')
    rows = [r for r in csv.DictReader(io.StringIO(source)) if r['method'] != 'Ours Stage1 (reference)']
    methods = ['Standard', 'LoRA TTA', 'DINOv2 concat-MLP', 'TTT', 'Ours']
    lookup = {r['method']: r for r in rows}
    labels = {'Standard': 'Standard Pooled WM', 'Ours': 'Ours (Stage2)'}
    tasks = [
        ('door', 'Door close', '100 test queries / 10 envs', 'action_pair_overlap_pct'),
        ('ball', 'Ball friction', '80 test queries / 8 envs', 'action_pair_overlap_pct'),
        ('stick', 'Stick balance', '45 test queries; Action: 36*', 'action_metric'),
        ('soft', 'Soft pull', '12 test queries; Onset: 11**', 'action_onset_success6_pct'),
    ]
    chart = Chart()
    chart.text(WIDTH / 2, 62, 'Real-world benchmark: detailed metrics', 40, True, 'center')
    chart.text(WIDTH / 2, 116, 'Current main-table results; task-specific evaluation protocols retained', 25, align='center')
    chart.line(170, 4)
    positions = [72, 392, 845, 1028, 1220, 1420, 1605, 1905]
    headers = ['Task', 'Method', 'PSNR (up)', 'SSIM (up)', 'LPIPS (down)', 'ADE (down)', 'FDE (down)', 'Action Score (up)']
    for i, (x, label) in enumerate(zip(positions, headers)):
        chart.text(x, 208, label, 23, True, 'left' if i < 2 else 'center')
    chart.line(245, 2)
    top = 245
    detailed_rows = []
    for task, title, cohort, action_suffix in tasks:
        for index, method in enumerate(methods):
            source_row = lookup[method]
            y = top + 39 + 67 * index
            ours = method == 'Ours'
            if index == 0:
                chart.text(positions[0], y, title, 25)
                chart.text(positions[0], y + 27, cohort, 16, color='#555')
            chart.text(positions[1], y, labels.get(method, method), 24, ours)
            values = []
            exported = dict(task=task, method=method)
            for suffix, digits, unit in [('psnr', 2, ''), ('ssim', 4, ''), ('lpips', 5, ''),
                                         ('ade_px', 2, ' px'), ('fde_px', 2, ' px')]:
                raw = source_row.get(task + '_' + suffix, '')
                exported[suffix] = raw
                values.append(f'{float(raw):.{digits}f}' + unit if raw else '-')
            raw_action = source_row.get(task + '_' + action_suffix, '') if action_suffix else ''
            values.append(f'{float(raw_action):.2f}%' if raw_action else '-')
            exported['action_score_pct'] = raw_action
            exported['action_definition'] = ('sliding_onset_success6deg_posthoc_gtpositive11' if task == 'soft' else
                                             'predicted_balanced_precision_posthoc36' if task == 'stick' else
                                             'first_closing_level_exact' if task == 'door' else
                                             'first_reaching_level_exact_marker_minus30px')
            for x, value in zip(positions[2:], values):
                chart.text(x, y, value, 24, ours, 'center', '#111' if value not in ('-', 'N/A') else '#777')
            detailed_rows.append(exported)
        top += 358
        chart.line(top, 3 if task == 'soft' else 2)
    footnotes = [
        'PSNR / SSIM: higher is better. LPIPS / ADE / FDE: lower is better. ADE and FDE are in pixels.',
        'Door Action: exact first-closing level. The unclosable environment is excluded.',
        'Ball Action: exact first level reaching the blue marker minus 30 px.',
        '* Stick Action: precision on post-hoc seed20260927, 36 cases (one balanced + three unbalanced per environment).',
        'Stick image / ADE: full 45-query set. FDE: 33 valid queries under the common GT / Standard / Ours / LoRA mask.',
        'Soft: six shared environments, 12 test queries; Onset Success@6 degrees uses 11 GT-positive queries.',
        '** Soft onset: >8px x motion for 3 frames. The 6-degree tolerance was selected post hoc after threshold comparison.',
        'Ours = selected sim-LR Stage2; Stage1 is not displayed.',
        'Bold denotes Ours, not a best-score claim. Missing / unreported values: -; not applicable: N/A.',
        'Tracking-derived metrics remain provisional. The selected Stick Action subset is not the full-test benchmark.',
    ]
    for index, text in enumerate(footnotes):
        chart.text(72, top + 42 + index * 35, text, 21, color='#444')
    stem = 'real97_main_results_detailed'
    chart.save(stem)
    with (OUT / (stem + '.csv')).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detailed_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(detailed_rows)
    print(json.dumps(dict(svg=str(OUT / (stem + '.svg')), png=str(OUT / (stem + '.png')),
                          csv=str(OUT / (stem + '.csv')), rows=len(detailed_rows), original_main_table_modified=False)))


if __name__ == '__main__':
    main()
