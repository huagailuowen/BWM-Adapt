#!/usr/bin/env python3
"""Chart-only main results, with explicit task-specific evaluation populations."""
import csv
import io
import json
import plot_real97_main_results as renderer


def main():
    out = renderer.OUT
    rows = list(csv.DictReader(io.StringIO(
        (out / 'real97_all_methods_main_table.csv').read_text().replace('\\r\\n', '\n'))))
    rows = [row for row in rows if row['method'] != 'Ours Stage1 (reference)']
    renderer.WIDTH, renderer.HEIGHT = 2720, 710
    c = renderer.Canvas()
    c.svg[2] = '<desc>Soft: shared six environments, twelve test queries; onset Success@6 degrees uses eleven GT-onset-positive queries, with an 8-pixel persistent-motion rule. The 6-degree tolerance was selected post hoc. Stick image/object: full 45 test queries; DINO/TTT FDE each has 35 valid queries and current Ours has 36. Tracking masks differ across source evaluations. Stick action: post-hoc seed20260927 subset of 36 queries. Ours denotes selected sim-LR Stage2 with task-specific cluster/family/global-mean initialization; Stage1 is not displayed. Missing values are not zero.</desc>'
    definitions = [
        ('door', 'DOOR CLOSE', '100 test queries; action: 9 environments',
         [('LPIPS', 'lpips', 5), ('ADE', 'ade_px', 2), ('FDE', 'fde_px', 2), ('Action', 'action_pair_overlap_pct', 2)]),
        ('ball', 'BALL FRICTION', '80 test queries / 8 environments',
         [('LPIPS', 'lpips', 5), ('ADE', 'ade_px', 2), ('FDE', 'fde_px', 2), ('Action', 'action_pair_overlap_pct', 2)]),
        ('stick', 'STICK BALANCE', 'Image/object: 45 queries; action: 36*',
         [('LPIPS', 'lpips', 5), ('ADE', 'ade_px', 2), ('FDE', 'fde_px', 2), ('Action', 'action_metric', 2)]),
        ('soft', 'SOFT PULL', '6 environments / 12 queries; onset: 11**',
         [('LPIPS', 'lpips', 5), ('ADE', 'ade_px', 2), ('FDE', 'fde_px', 2), ('Onset@6\u00b0', 'action_onset_success6_pct', 2)]),
    ]
    left, start, col = 30, 340, 144
    right = start + 16*col
    c.line(left, 22, right, 22, renderer.INK, 2)
    c.text(left+16, 143, 'METHOD', 21, renderer.MUTED, True)
    positions = []
    for task, title, subtitle, columns in definitions:
        width = col*len(columns)
        c.text(start+width/2, 63, title, 25, renderer.INK, True, 'center')
        c.text(start+width/2, 96, subtitle, 16, renderer.MUTED, align='center')
        for i, (label, suffix, digits) in enumerate(columns):
            x = start + col*(i+.5)
            is_action = label == 'Action' or suffix == 'action_onset_success6_pct'
            c.text(x, 143, label+(' \u2191' if is_action else ' \u2193'),
                   20, renderer.MUTED, True, 'center')
            positions.append((task+'_'+suffix, digits, x, is_action))
        c.line(start+12, 165, start+width-12, 165, renderer.RULE, 2)
        start += width
    c.line(left, 165, 325, 165, renderer.RULE, 2)
    labels = {'DINOv2 concat-MLP': 'DINOv2', 'Ours Stage1 (reference)': 'Ours Stage1 (ref.)'}
    comparison_rows = [r for r in rows if r['method'] != 'Ours Stage1 (reference)']
    for index, row in enumerate(rows):
        y = 215 + 72*index
        ours = row['method']=='Ours'
        reference = row['method']=='Ours Stage1 (reference)'
        if ours:
            c.rect(left, y-35, right-left, 66, renderer.PALE, 7)
            c.rect(left, y-35, 5, 66, renderer.ACCENT)
        if reference:
            c.line(left, y-35, right, y-35, renderer.RULE, 1)
        c.text(left+16, y+8, labels.get(row['method'],row['method']), 23 if reference else 27,
               renderer.ACCENT if ours else renderer.INK, ours)
        for key, digits, x, action in positions:
            if not row.get(key):
                c.text(x, y+8, '\u2013', 27, '#A0ACA9', align='center')
                continue
            value = float(row[key])
            pool = [float(r[key]) for r in comparison_rows if r.get(key)]
            best = not reference and bool(pool) and abs(value-(max(pool) if action else min(pool)))<1e-10
            c.text(x, y+8, f'{value:.{digits}f}'+('%' if action else ''), 25,
                   renderer.ACCENT if best else renderer.INK, best, 'center')
        if index < len(rows)-1:
            c.line(left+12, y+35, right-12, y+35, '#E7ECE9', 1)
    bottom = 215+72*(len(rows)-1)+41
    c.line(left, bottom, right, bottom, renderer.INK, 2)
    c.text(left, bottom+32, '* Stick action: post-hoc selected seed 20260927, 1 balanced + 3 unbalanced per environment; provisional.', 18, renderer.MUTED)
    c.text(left, bottom+60, '** Soft onset: 8px motion for 3 frames; Success@6 degrees on 11 GT-positive queries. Tolerance selected post hoc; provisional.', 18, renderer.MUTED)
    c.text(left, bottom+88, 'ADE/FDE: pixels. Door action: first-close level (exact 1, +/-1 level 0.5). Ball action: pair overlap.', 18, renderer.MUTED)
    c.text(left, bottom+116, 'Ours: selected sim-LR inference. Tracking-valid masks differ across methods; coverage and source protocols are recorded in README.', 18, renderer.MUTED)
    svg = out / 'real97_main_results.svg'
    png = out / 'real97_main_results.png'
    svg.write_text('\n'.join(c.svg+['</svg>'])+'\n')
    c.image.save(png, dpi=(300,300))
    print(json.dumps({'svg':str(svg),'png':str(png),'methods':len(rows)}))


if __name__ == '__main__':
    main()
