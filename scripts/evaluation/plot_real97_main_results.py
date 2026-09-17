#!/usr/bin/env python3
"""Render the frozen Real97 table as editable SVG and presentation-resolution PNG."""
import base64
import csv
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/real97_all_methods_main_table_v1'
WIDTH, HEIGHT, SCALE = 2400, 1220, 2
INK = '#182D35'
MUTED = '#687A80'
ACCENT = '#087D78'
PAPER = '#FAFBF9'
RULE = '#D5DEDA'
PALE = '#E8F3EF'


class Canvas:
    def __init__(self):
        self.image = Image.new('RGB', (WIDTH*SCALE, HEIGHT*SCALE), PAPER)
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = {}
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
                    '<title>Real-world prediction: four-task results</title>',
                    '<desc>Held-out test results for Door close, Ball friction, Stick balance, and Soft pull. Stick is pending. A matched-six-environment Soft comparison is shown separately.</desc>',
                    f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{PAPER}"/>']

    def font(self, size, bold=False, serif=False):
        key = (size, bold, serif)
        if key not in self.fonts:
            family = 'DejaVuSerif' if serif else 'DejaVuSans'
            name = family + ('-Bold' if bold else '') + '.ttf'
            self.fonts[key] = ImageFont.truetype(name, round(size*SCALE))
        return self.fonts[key]

    def rect(self, x, y, width, height, fill, radius=0):
        box = tuple(round(v*SCALE) for v in (x, y, x+width, y+height))
        self.draw.rounded_rectangle(box, radius=round(radius*SCALE), fill=fill)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}"/>')

    def line(self, x1, y1, x2, y2, fill=RULE, width=1):
        self.draw.line(tuple(round(v*SCALE) for v in (x1, y1, x2, y2)), fill=fill, width=max(1,round(width*SCALE)))
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{fill}" stroke-width="{width}"/>')

    def text(self, x, y, value, size=24, fill=INK, bold=False, align='left', serif=False):
        value = str(value)
        font = self.font(size, bold, serif)
        anchor = {'left':'ls', 'center':'ms', 'right':'rs'}[align]
        self.draw.text((round(x*SCALE), round(y*SCALE)), value, font=font, fill=fill, anchor=anchor)
        family = 'DejaVu Serif, Georgia, serif' if serif else 'DejaVu Sans, sans-serif'
        weight = '700' if bold else '400'
        svg_anchor = {'left':'start', 'center':'middle', 'right':'end'}[align]
        self.svg.append(f'<text x="{x}" y="{y}" text-anchor="{svg_anchor}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}">{html.escape(value)}</text>')


def main():
    with (OUT / 'real97_all_methods_main_table.csv').open(newline='') as handle:
        records = list(csv.DictReader(handle))
    methods = [row['method'] for row in records]
    names = {'Standard':'Standard', 'LoRA TTA':'LoRA TTA',
             'DINOv2 concat-MLP':'DINOv2', 'TTT':'TTT', 'Ours':'Ours'}
    down, up, missing = '\u2193', '\u2191', '\u2013'
    definitions = [
        ('door', 'DOOR CLOSE', '10 environments / 100 test queries',
         [('LPIPS '+down,'lpips',5),('ADE '+down,'ade_px',2),('FDE '+down,'fde_px',2),('Action '+up,'action_pair_overlap_pct',2)]),
        ('ball', 'BALL FRICTION', '8 environments / 80 test queries',
         [('LPIPS '+down,'lpips',5),('ADE '+down,'ade_px',2),('FDE '+down,'fde_px',2),('Action '+up,'action_pair_overlap_pct',2)]),
        ('stick', 'STICK BALANCE', 'Results pending',
         [('LPIPS '+down,'lpips',5),('Object '+down,'object_metric',2),('Action '+up,'action_metric',2)]),
        ('soft', 'SOFT PULL', '9 environments / 18 test queries',
         [('LPIPS '+down,'lpips',5),('ADE '+down,'ade_px',2),('FDE '+down,'fde_px',2)]),
    ]
    c = Canvas()
    c.rect(66, 61, 8, 94, ACCENT)
    c.text(101, 83, 'REAL-WORLD EXPERIMENTS', 21, ACCENT, True)
    c.text(98, 144, 'Predicting physical outcomes', 51, INK, serif=True)
    c.text(99, 184, 'Held-out test episodes  /  Support from training episodes  /  Frozen metric release', 23, MUTED)
    c.text(2334, 81, 'CORE METRICS', 20, MUTED, True, 'right')
    c.text(2334, 116, 'Lower error. Better action selection.', 21, MUTED, align='right')
    c.text(2334, 151, 'Best available value in bold', 20, MUTED, align='right')

    left, method_width, column_width = 66, 260, 139
    first = left + method_width
    c.line(left, 223, 2334, 223, INK, 2)
    c.text(left+20, 337, 'METHOD', 21, MUTED, True)
    positions = []
    start = first
    for task, title, subtitle, columns in definitions:
        width = column_width * len(columns)
        center = start + width/2
        c.text(center, 267, title, 25, INK, True, 'center')
        c.text(center, 297, subtitle, 18, MUTED, align='center')
        if task == 'door':
            c.text(center, 317, 'Action scoring: 9 closable environments', 15, MUTED, align='center')
        for j,(label,suffix,digits) in enumerate(columns):
            x = start + column_width*(j+0.5)
            c.text(x, 348, label, 20, MUTED, True, 'center')
            positions.append((task, suffix, digits, x))
        c.line(start+14, 364, start+width-14, 364, RULE, 2)
        start += width
    c.line(left, 364, first-12, 364, RULE, 2)

    row_centers = [410, 484, 558, 632, 706]
    for index,row in enumerate(records):
        cy = row_centers[index]
        if row['method'] == 'Ours':
            c.rect(left, cy-40, 2268, 69, PALE, 7)
            c.rect(left, cy-40, 5, 69, ACCENT)
        c.text(left+20, cy+9, names[row['method']], 27,
               ACCENT if row['method']=='Ours' else INK, row['method']=='Ours')
        for task,suffix,digits,x in positions:
            key = task+'_'+suffix
            raw = row.get(key, '')
            if not raw:
                c.text(x, cy+8, missing, 27, '#A0ACA9', align='center')
                continue
            value = float(raw)
            pool = [float(r[key]) for r in records if r.get(key)]
            optimum = max(pool) if 'action' in suffix else min(pool)
            best = abs(value-optimum) < 1e-10
            display = f'{value:.{digits}f}'
            if 'action' in suffix:
                display += '%'
            c.text(x, cy+8, display, 25, ACCENT if best else INK, best, 'center')
        if index < 4:
            c.line(left+16, cy+37, 2318, cy+37, '#E7ECE9', 1)
    c.line(left, 749, 2334, 749, INK, 2)
    c.text(left, 784, 'ADE / FDE in pixels. Action = preferred-level pair overlap (0, 0.5, or 1), not binary success.', 20, MUTED)
    c.text(left, 815, 'Soft above: all 9 environments. Standard did not train on 4L, 7R or 6M; the matched-coverage comparison is below.', 20, MUTED)

    # Frozen shared-six object means from the accepted family-mean comparison.
    # These are previously computed scores, not a new evaluation or refit.
    shared_lpips = json.loads((OUT / 'soft_lpips/summary.json').read_text())['test_shared_training_environments']
    shared = {
        'Standard': {'lpips':shared_lpips['standard'], 'ade':32.78484249, 'fde':39.41445094},
        'Ours': {'lpips':shared_lpips['ours'], 'ade':18.09544777, 'fde':22.37870640},
    }
    c.rect(left, 853, 2268, 246, '#EFF3F0', 10)
    c.text(left+30, 896, 'SOFT PULL / MATCHED TRAINING COVERAGE', 23, ACCENT, True)
    c.text(left+30, 938, 'Same 6 environments / same 12 held-out queries', 24, INK)
    c.text(left+30, 975, '1L, 2L, 1R, 5R, 2M, 8', 22, MUTED)
    ade_gain = 100*(1-shared['Ours']['ade']/shared['Standard']['ade'])
    fde_gain = 100*(1-shared['Ours']['fde']/shared['Standard']['fde'])
    c.text(left+30, 1045, f'ADE  -{ade_gain:.1f}%     FDE  -{fde_gain:.1f}%', 30, ACCENT, True)
    c.text(1200, 905, 'METHOD', 20, MUTED, True)
    for x,label in [(1600,'LPIPS '+down),(1860,'ADE '+down),(2140,'FDE '+down)]:
        c.text(x, 905, label, 21, MUTED, True, 'center')
    c.line(1195, 925, 2300, 925, '#CCD8D1', 1)
    for name,cy in [('Standard',976),('Ours',1050)]:
        bold = name=='Ours'
        color = ACCENT if bold else INK
        c.text(1200, cy, name, 28, color, bold)
        c.text(1600, cy, f"{shared[name]['lpips']:.5f}", 28, color, bold, 'center')
        c.text(1860, cy, f"{shared[name]['ade']:.2f}", 28, color, bold, 'center')
        c.text(2140, cy, f"{shared[name]['fde']:.2f}", 28, color, bold, 'center')
    c.text(left, 1141, 'Ours = Stage2 with the recorded environment/family-informed initialization. Information conditions and training budgets differ.', 19, MUTED)
    c.text(left, 1172, 'Tracking scores remain provisional. Means shown without confidence intervals. PSNR/SSIM remain in the complete CSV/Markdown tables.', 19, MUTED)
    c.text(2334, 1200, 'REAL97  /  2026-09-16', 16, MUTED, align='right')

    svg_path = OUT / 'real97_main_results.svg'
    png_path = OUT / 'real97_main_results.png'
    svg_path.write_text('\n'.join(c.svg + ['</svg>']) + '\n')
    c.image.save(png_path, dpi=(300,300))
    preview = c.image.resize((1440,732), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, format='JPEG', quality=85)
    print(json.dumps({'svg':str(svg_path),'png':str(png_path),
                      'preview_jpeg_base64':base64.b64encode(buffer.getvalue()).decode('ascii')}))


if __name__ == '__main__':
    main()
