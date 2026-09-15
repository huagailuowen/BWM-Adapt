#!/usr/bin/env python3
"""Render final Soft training/inference PCA with the established environment palette."""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/infer_real97_soft_balanced9_family_mean_static_20260914_v1'
COLORS = {
    'soft-1l': '#cc3d3d', 'soft-2l': '#cc9c3d', 'soft-4l': '#9ccc3d',
    'soft-1r': '#3dcc3d', 'soft-7r': '#3dcc9c', 'soft-5r': '#3d9ccc',
    'soft-2m': '#3d3dcc', 'soft-6m': '#9c3dcc', 'soft-8': '#cc3d9c',
}


class Canvas:
    def __init__(self, width, height):
        self.width, self.height, self.scale = width, height, 2
        self.image = Image.new('RGB', (width * 2, height * 2), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = {}
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
        ]

    def font(self, size):
        size = int(size * self.scale)
        if size not in self.fonts:
            try:
                self.fonts[size] = ImageFont.truetype('DejaVuSans.ttf', size)
            except OSError:
                self.fonts[size] = ImageFont.load_default()
        return self.fonts[size]

    def text(self, x, y, value, size=14, color='#263238', align='left'):
        anchor = {'left': 'lm', 'center': 'mm', 'right': 'rm'}[align]
        svg_anchor = {'left': 'start', 'center': 'middle', 'right': 'end'}[align]
        self.draw.text((x * 2, y * 2), value, font=self.font(size), fill=color, anchor=anchor)
        self.svg.append(
            f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{size}" '
            f'fill="{color}" text-anchor="{svg_anchor}" dominant-baseline="middle">{escape(value)}</text>')

    def line(self, x0, y0, x1, y1, color='#dce1e5', width=1):
        self.draw.line((x0 * 2, y0 * 2, x1 * 2, y1 * 2), fill=color, width=max(1, round(width * 2)))
        self.svg.append(f'<path d="M{x0},{y0} L{x1},{y1}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def marker(self, x, y, kind, fill, radius=6, title=''):
        if kind == 'circle':
            self.draw.ellipse(((x-radius)*2, (y-radius)*2, (x+radius)*2, (y+radius)*2),
                              fill=fill, outline='#171717', width=2)
            self.svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="#171717" stroke-width="1"><title>{escape(title)}</title></circle>')
            return
        if kind == 'triangle':
            points = [(x, y-radius), (x-radius*.87, y+radius*.6), (x+radius*.87, y+radius*.6)]
        else:
            points = [(x, y-radius), (x+radius, y), (x, y+radius), (x-radius, y)]
        self.draw.polygon([(round(a*2), round(b*2)) for a,b in points], fill=fill)
        self.draw.line([(round(a*2), round(b*2)) for a,b in points+[points[0]]],
                       fill='#171717', width=2)
        coordinates = ' '.join(f'{a},{b}' for a,b in points)
        self.svg.append(f'<polygon points="{coordinates}" fill="{fill or "none"}" stroke="#171717" stroke-width="1"><title>{escape(title)}</title></polygon>')

    def save(self, stem):
        svg_path = OUTPUT / (stem + '.svg')
        png_path = OUTPUT / (stem + '.png')
        png = io.BytesIO()
        self.image.resize((self.width, self.height), Image.Resampling.LANCZOS).save(png, format='PNG')
        data = png.getvalue()
        with svg_path.open('x') as handle:
            handle.write('\n'.join(self.svg + ['</svg>']) + '\n')
        with png_path.open('xb') as handle:
            handle.write(data)
        return dict(svg=str(svg_path), png=str(png_path),
                    image='data:image/png;base64,' + base64.b64encode(data).decode())


def read(path):
    return json.loads(path.read_text())


def axes(canvas, box, points, xlabel, ylabel, decimals=2, padding=.12):
    left, top, width, height = box
    points = np.asarray(points)
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high-low, 0.001)
    midpoint = (low+high)/2
    low = midpoint-span*(.5+padding)
    high = midpoint+span*(.5+padding)
    span = high-low
    def project(point):
        return left+width*(point[0]-low[0])/span[0], top+height*(high[1]-point[1])/span[1]
    for value in np.linspace(low[0], high[0], 5):
        x, _ = project((value, low[1]))
        canvas.line(x, top, x, top+height)
        canvas.text(x, top+height+18, f'{value:.{decimals}f}', 11, '#59636d', 'center')
    for value in np.linspace(low[1], high[1], 5):
        _, y = project((low[0], value))
        canvas.line(left, y, left+width, y)
        canvas.text(left-10, y, f'{value:.{decimals}f}', 11, '#59636d', 'right')
    canvas.line(left, top+height, left+width, top+height, '#7b858e')
    canvas.line(left, top, left, top+height, '#7b858e')
    canvas.text(left+width/2, top+height+48, xlabel, 14, align='center')
    canvas.text(left, top-20, ylabel, 14)
    return project


def main():
    table = read(OUTPUT / 'input_manifest/context_table.json')
    initialization = read(OUTPUT / 'group_initialization.json')
    rows = table['records']
    values = np.asarray([row['context'] for row in rows], dtype=np.float64).reshape(len(rows), -1)
    center = values.mean(axis=0)
    _, singular, axes_matrix = np.linalg.svd(values-center, full_matrices=False)
    basis = axes_matrix[:2]
    explained = singular[:2]**2 / np.sum(singular**2)
    training = (values-center) @ basis.T
    contexts = {name: read(OUTPUT / 'contexts' / (name + '.json')) for name in COLORS}
    inference = {name: (np.asarray(state['context']).reshape(-1)-center) @ basis.T
                 for name,state in contexts.items()}
    starts = {group: (np.asarray(item['context']).reshape(-1)-center) @ basis.T
              for group,item in initialization['groups'].items()}
    x_label, y_label = f'PC1 ({100*explained[0]:.1f}%)', f'PC2 ({100*explained[1]:.1f}%)'

    global_canvas = Canvas(1160, 760)
    global_canvas.text(65, 36, 'Soft pull: final training and inference latents', 24)
    global_canvas.text(65, 67, 'Step 5500 | family-mean initialization | 1R belongs to R', 14, '#59636d')
    all_points = np.vstack([training, *[xy.reshape(1,2) for xy in inference.values()]])
    project = axes(global_canvas, (90, 125, 785, 475), all_points, x_label, y_label)
    for row, point in zip(rows, training):
        x, y = project(point)
        global_canvas.marker(x, y, 'circle', COLORS[row['environment']], 5.5,
                             f"{row['environment']} training time, replica {row['replica']}")
    for name, point in inference.items():
        x, y = project(point)
        global_canvas.marker(x, y, 'triangle', COLORS[name], 7.5, f'{name} inference time')
    global_canvas.text(920, 131, 'Environment', 16)
    for i, (name, color) in enumerate(COLORS.items()):
        global_canvas.marker(928, 164+i*28, 'circle', color, 5)
        global_canvas.text(944, 164+i*28, name.removeprefix('soft-').upper(), 14)
    global_canvas.marker(928, 446, 'circle', '#929aa1', 6)
    global_canvas.text(944, 446, 'Training time', 13)
    global_canvas.marker(928, 477, 'triangle', '#929aa1', 7)
    global_canvas.text(944, 477, 'Inference time', 13)
    global_canvas.text(65, 694, '18 training latents; 9 support-adapted latents. Colors match the previous figures.', 13, '#59636d')
    global_canvas.text(65, 721, 'The same adapted latent serves both train/test queries. No trajectories, jitter, or query-GT fitting.', 12, '#59636d')
    global_result = global_canvas.save('final_training_inference_Z_pca')

    zoom = Canvas(1320, 540)
    zoom.text(65, 33, 'Inference-time detail: same PCA plane, expanded axes', 23)
    zoom.text(65, 64, 'No refitting or point displacement. Hollow diamond: family-mean starting point.', 14, '#59636d')
    for col, group in enumerate(('L', 'R', '8')):
        names = initialization['groups'][group]['environments']
        points = np.vstack([*[inference[name] for name in names], starts[group]])
        left = 82+col*435
        zoom.text(left+143, 103, f'{group} group', 17, align='center')
        project_local = axes(zoom, (left, 155, 290, 225), points, x_label, y_label, decimals=4, padding=.32)
        x,y = project_local(starts[group])
        zoom.marker(x,y,'diamond',None,8,f'{group} starting mean')
        for name in names:
            x,y = project_local(inference[name])
            zoom.marker(x,y,'triangle',COLORS[name],7.5,name+' inference time')
    for i,(name,color) in enumerate(COLORS.items()):
        x=78+i*137
        zoom.marker(x,484,'triangle',color,6)
        zoom.text(x+13,484,name.removeprefix('soft-').upper(),13)
    zoom.text(65,519,'Each panel has independent axis limits, but all coordinates use the original 18-row training PCA basis.',12,'#59636d')
    zoom_result = zoom.save('final_inference_Z_pca_zoom')
    coordinates = dict(
        source=str(OUTPUT),model_step=5500,table_step=5500,
        basis_fit='18 training rows only; inference excluded from fitting',
        center=center.tolist(),components=basis.tolist(),explained_variance_ratio=explained.tolist(),
        colors=COLORS,
        training=[dict(environment=row['environment'],replica=row['replica'],xy=point.tolist()) for row,point in zip(rows,training)],
        inference={name:dict(xy=point.tolist(),family=initialization['environment_to_group'][name]) for name,point in inference.items()},
        initial_means={name:point.tolist() for name,point in starts.items()},
        query_split_note='Adaptation uses training supports only; one final latent per environment is shared by train/test queries.',
        point_jitter=False,
    )
    with (OUTPUT / 'final_Z_pca_coordinates.json').open('x') as handle:
        json.dump(coordinates,handle,indent=2)
        handle.write('\n')
    print(json.dumps(dict(global_plot=global_result,zoom_plot=zoom_result,
                          explained_variance_ratio=explained.tolist())))


if __name__ == '__main__':
    main()

