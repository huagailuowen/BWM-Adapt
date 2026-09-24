#!/usr/bin/env python3
"""Render the formal per-task PCA projections as a dependency-free SVG."""
from __future__ import annotations
import argparse, colorsys, csv, html
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = [('event80','Push-box friction'),('gravity','Gravity'),('mass_collision','Mass collision'),
         ('light_switch','Light switch'),('mass_balance','Mass balance'),('mass_friction','Mass + friction')]
W,H=1450,880; PW,PH=430,315; LEFT,TOP,GX,GY=75,115,45,65

def tint(v,lo,hi):
    t=.5 if hi==lo else max(0,min(1,(v-lo)/(hi-lo)))
    stops=[(0,(42,72,200)),(.34,(30,190,210)),(.67,(245,205,55)),(1,(210,45,45))]
    for (a,ca),(b,cb) in zip(stops,stops[1:]):
        if t<=b:
            u=(t-a)/(b-a); rgb=tuple(round(x+u*(y-x)) for x,y in zip(ca,cb)); return '#%02x%02x%02x'%rgb
    return '#d22d2d'

def darken(color, factor=.72):
    red=int(color[1:3],16); green=int(color[3:5],16); blue=int(color[5:7],16)
    return '#%02x%02x%02x'%(round(red*factor),round(green*factor),round(blue*factor))

def row_color(row):
    scheme=row.get('color_scheme','continuous_absolute')
    if scheme=='categorical_environment':
        color={
            'neither': '#778391',
            'red_only': '#d54b4b',
            'blue_only': '#3979c9',
            'both': '#2c9a68',
        }[row['color_primary_value']]
    elif scheme=='bivariate_ordinal':
        primary_count=max(int(row['color_primary_count'])-1,1)
        secondary_count=max(int(row['color_secondary_count'])-1,1)
        friction_rank=float(row['color_primary_rank'])/primary_count
        mass_rank=float(row['color_secondary_rank'])/secondary_count
        hue=(225.0-215.0*friction_rank)/360.0
        lightness=.74-.34*mass_rank
        red,green,blue=colorsys.hls_to_rgb(hue,lightness,.72)
        color='#%02x%02x%02x'%(round(255*red),round(255*green),round(255*blue))
    elif scheme=='ordinal':
        color=tint(float(row['color_primary_rank']),0,max(int(row['color_primary_count'])-1,1))
    else:
        color=tint(float(row['color_primary_value']),float(row['color_primary_min']),float(row['color_primary_max']))
    return darken(color) if row.get('color_tone')=='dark' else color

def txt(x,y,s,size=12,weight=400,anchor='start',fill='#202833'):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{html.escape(s)}</text>'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-csv',default='results/sim_all_methods_main_table_v1/ours_formal_train_test_latents_pca_internal.csv'); p.add_argument('--output-svg',default='results/sim_all_methods_main_table_v1/ours_formal_train_test_latents_pca.svg'); a=p.parse_args()
    with (ROOT/a.input_csv).open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">','<rect width="100%" height="100%" fill="white"/>',txt(55,36,'Ours: training-time environment codes and test-time latent adaptation',21,700),txt(55,61,'Per-task PCA is fitted on active training codes only; test-time points are projected without refitting.',12,fill='#5b6673')]
    for n,(task,title) in enumerate(TASKS):
        rr=[r for r in rows if r['task']==task]; tr=[r for r in rr if r['phase']=='training_time']; inf=[r for r in rr if r['phase']=='inference_time']
        paths=defaultdict(list)
        for r in inf: paths[r['support_indices_json']].append(r)
        for v in paths.values(): v.sort(key=lambda r:int(r['inner_step']))
        x0=LEFT+(n%3)*(PW+GX); y0=TOP+(n//3)*(PH+GY); pl,pt=x0+56,y0+50; pw,ph=PW-75,PH-92
        xs=[float(r['PC1']) for r in rr]; ys=[float(r['PC2']) for r in rr]; lo,hi=min(xs),max(xs); bot,top=max(min(ys),-1e100),min(max(ys),1e100)
        xp=max((hi-lo)*.1,.1); yp=max((top-bot)*.1,.1); lo-=xp; hi+=xp; bot-=yp; top+=yp
        def xy(r): return pl+(float(r['PC1'])-lo)/(hi-lo)*pw, pt+ph-(float(r['PC2'])-bot)/(top-bot)*ph
        s += [f'<rect x="{x0}" y="{y0}" width="{PW}" height="{PH}" rx="5" fill="#fbfcfd" stroke="#c9d0d9"/>',txt(x0+14,y0+22,title,15,700),txt(x0+14,y0+41,f'{len(tr)} train codes, {len(paths)} test-time episodes',10,fill='#63707d')]
        for i in range(6):
            xx=pl+i*pw/5; yy=pt+i*ph/5; s += [f'<line x1="{xx:.1f}" y1="{pt}" x2="{xx:.1f}" y2="{pt+ph}" stroke="#e1e5ea" stroke-width=".7"/>',f'<line x1="{pl}" y1="{yy:.1f}" x2="{pl+pw}" y2="{yy:.1f}" stroke="#e1e5ea" stroke-width=".7"/>']
        s.append(f'<rect x="{pl}" y="{pt}" width="{pw}" height="{ph}" fill="none" stroke="#3d4650" stroke-width=".8"/>')
        for v in paths.values():
            pts=[xy(r) for r in v]; c=row_color(v[-1]); q=' '.join(f'{x:.1f},{y:.1f}' for x,y in pts); sx,sy=pts[0]; fx,fy=pts[-1]; edge='#111820' if v[-1]['context_group_domain']=='ood' else 'white'
            s += [f'<polyline points="{q}" fill="none" stroke="{c}" stroke-width="1.25" stroke-opacity=".62"/>',f'<path d="M {sx-3:.1f},{sy-3:.1f} L {sx+3:.1f},{sy+3:.1f} M {sx+3:.1f},{sy-3:.1f} L {sx-3:.1f},{sy+3:.1f}" stroke="{c}" stroke-width="1.4"/>',f'<polygon points="{fx:.1f},{fy-6:.1f} {fx-5.5:.1f},{fy+4.5:.1f} {fx+5.5:.1f},{fy+4.5:.1f}" fill="{c}" stroke="{edge}" stroke-width="1.2"/>']
        for r in tr:
            x,y=xy(r); c=row_color(r); s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.4" fill="{c}" stroke="#17202a" stroke-width=".8"/>')
        v1=100*float(rr[0]['PC1_explained_variance_ratio']); v2=100*float(rr[0]['PC2_explained_variance_ratio']); s += [txt(pl+pw/2,y0+PH-12,f'PC1 ({v1:.1f}% train variance)',10,anchor='middle'),f'<text x="{x0+16}" y="{pt+ph/2}" font-family="DejaVu Sans, sans-serif" font-size="10" text-anchor="middle" fill="#202833" transform="rotate(-90 {x0+16} {pt+ph/2})">PC2 ({v2:.1f}% train variance)</text>']
    y=842; s += ['<circle cx="155" cy="837" r="5" fill="#2eb9cf" stroke="#17202a"/>',txt(168,y,'Training-time code',11),'<path d="M 342,833 L 350,841 M 350,833 L 342,841" stroke="#59636f" stroke-width="1.5"/>',txt(359,y,'Test-time initial Z (darker matched color)',11),'<line x1="590" y1="837" x2="616" y2="837" stroke="#59636f" stroke-width="1.5"/>',txt(625,y,'Adaptation trajectory',11),'<polygon points="820,831 814,842 826,842" fill="#b1762e" stroke="white"/>',txt(834,y,'Final Z, active group',11),'<polygon points="1055,831 1049,842 1061,842" fill="#b1762e" stroke="black"/>',txt(1069,y,'Final Z, held-out group',11),txt(1385,y,'colors referenced to active training physics only',10,anchor='end',fill='#63707d'),'</svg>']
    out=ROOT/a.output_svg; out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_suffix(out.suffix+'.tmp'); tmp.write_text('\n'.join(s)+'\n',encoding='utf-8'); tmp.replace(out); print(f'[done] {out}')
if __name__=='__main__': main()
