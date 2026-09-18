"""Small, isolated helpers for the September-16 Stick comparison."""
import json
from html import escape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def plot_pca(table, environments, output, step, contexts=None):
    """Fit only training Z; identical plane for global and inference-only views."""
    records = table['records']
    values = np.asarray([r['context'] for r in records], float).reshape(len(records), -1)
    center = values.mean(0)
    _, singular, axes = np.linalg.svd(values - center, full_matrices=False)
    for i in range(2):
        if axes[i, np.argmax(abs(axes[i]))] < 0:
            axes[i] *= -1
    xy = (values - center) @ axes[:2].T
    variance = singular ** 2 / max(float((singular ** 2).sum()), 1e-30)
    index = {int(r['friction_mu']): i for i, r in enumerate(records)}
    colors = dict(zip(['stick-L4-R0', 'stick-L3-R0', 'stick-L2-R0', 'stick-L1-R0',
                      'stick-L0-R0', 'stick-L0-R1', 'stick-L0-R2', 'stick-L0-R3', 'stick-L0-R4'],
                     ['#2463eb', '#2173c4', '#1c8a89', '#25a346', '#7aa12e',
                      '#ce9f16', '#f59c0b', '#ed7714', '#dc2626']))
    contexts = contexts or {}
    endpoints = {name: (np.asarray(state['context']).reshape(-1) - center) @ axes[:2].T
                 for name, state in contexts.items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for zoom in ([False, True] if endpoints else [False]):
        cloud = np.stack(list(endpoints.values())) if zoom else np.vstack([xy, *[p[None] for p in endpoints.values()]])
        low, high = cloud.min(0), cloud.max(0)
        span = np.maximum(high - low, .05)
        low -= .13 * span
        high += .13 * span
        sx = lambda x: 90 + 820 * (float(x) - low[0]) / (high[0] - low[0])
        sy = lambda y: 650 - 500 * (float(y) - low[1]) / (high[1] - low[1])
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="760" viewBox="0 0 1240 760">',
                 '<rect width="100%" height="100%" fill="#fbfaf6"/>']
        def text(x, y, message, size=15, color='#25313a'):
            parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{size}" fill="{color}">{escape(message)}</text>')
        title = 'inference time (same PCA plane)' if zoom else ('training and inference time' if endpoints else 'training time')
        text(90, 45, f'Stick balance | {title} | model/table step {step}', 23)
        text(90, 82, f'{len(records)} active environments | PC1 + PC2 = {sum(variance[:2])*100:.2f}%', 16, '#746f66')
        parts.append('<rect x="90" y="150" width="820" height="500" fill="#fffdf8" stroke="#d8d2c7"/>')
        for i in range(6):
            a, b = low + (high - low) * i / 5
            parts.append(f'<path d="M{sx(a)},150V650 M90,{sy(b)}H910" stroke="#ded9ce" fill="none"/>')
            text(sx(a)-15, 675, f'{a:.2f}', 12)
            text(20, sy(b)+4, f'{b:.2f}', 12)
        text(90, 130, f'PC2 ({variance[1]*100:.2f}%)')
        text(400, 714, f'PC1 ({variance[0]*100:.2f}%)')
        for i, env in enumerate(environments):
            name = env['environment']
            color = colors[name]
            if not zoom:
                point = xy[index[int(env['environment_index'])]]
                x, y = sx(point[0]), sy(point[1])
                parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" stroke="#25313a" stroke-width=".6"><title>{escape(name)} training time</title></circle>')
                text(x+9, y+(17 if i % 2 else -9), name.removeprefix('stick-'), 12)
            if name in endpoints:
                x, y = sx(endpoints[name][0]), sy(endpoints[name][1])
                parts.append(f'<polygon points="{x},{y-8} {x-7},{y+6} {x+7},{y+6}" fill="{color}" stroke="black"><title>{escape(name)} inference time</title></polygon>')
            parts.append(f'<circle cx="950" cy="{171+i*33}" r="6" fill="{color}"/>')
            text(966, 176+i*33, name.removeprefix('stick-'))
        text(940, 535, 'Circles: training time', 14)
        if endpoints:
            text(940, 563, 'Triangles: inference time', 14)
        parts.append('</svg>')
        name = 'training_inference_Z_pca_inference_zoom' if zoom else ('training_inference_Z_pca' if endpoints else f'stick916_final_step{step}_training_pca')
        (output / (name + '.svg')).write_text('\n'.join(parts))
    write_json(output / ('training_inference_Z_pca_projection.json' if endpoints else f'stick916_final_step{step}_training_pca.json'),
               {'step': step, 'fit_on': 'training_table_only', 'center': center.tolist(),
                'components': axes[:2].tolist(), 'explained': variance.tolist(),
                'training_xy': xy.tolist(), 'inference_xy': {k: v.tolist() for k, v in endpoints.items()},
                'environments': environments})
