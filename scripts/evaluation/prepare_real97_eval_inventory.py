import csv, json, random
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np

import os
if not os.environ.get('SLURM_JOB_ID'):
 raise RuntimeError('Batch video contact sheets require a Slurm compute allocation.')

cv2.setNumThreads(1)
root=Path.cwd()
out=root/'outputs/evaluation_real97_dataset_audit_20260909'
out.mkdir(parents=True,exist_ok=True)
(out/'contact_sheets').mkdir(exist_ok=True)
datasets={
 'ball':'ball_friction_9_7_crop','door':'door_close_9_7',
 'stick':'stick_balance_9_7','soft':'soft_pull_9_7_crop_512x256',
}
all_rows=[]
summary={}
door_minimum={'door-0':1,'door-2u-1d':2,'door-3u':3,'door-2u-4d':4,
 'door-4d':3,'door-3u-4d':5,'door-6u-7d':7,'door-6u-8d':8,
 'door-12u-Half':9,'door-12u-Half-11d-Half':None}
for task,dataset in datasets.items():
 base=(root/'../../datasets_real'/dataset).resolve()
 split_lookup={}
 heldout=set()
 if task=='door':
  for line in (base/'episode_level_split_manifest.jsonl').read_text().splitlines():
   if line.strip():
    row=json.loads(line)
    split_lookup[(row['environment'],int(row['formal_episode_index']))]=row['split']
 else:
  split=json.loads((base/'TRAIN_TEST_SPLIT.json').read_text())
  if task=='ball':
   tests={(r['environment'],int(r['lerobot_episode_index'])) for r in split['test_episodes']}
   for envlevel,episodes in split['candidates_by_environment_level'].items():
    env=envlevel.split('/')[0]
    for episode in episodes:
     key=(env,int(episode)); split_lookup[key]='test' if key in tests else 'train'
  elif task=='stick':
   for env in split['environments']:
    for tag in ('train','test'):
     for episode in env[tag+'_episode_indices']:
      split_lookup[(env['environment'],int(episode))]=tag
  else:
   heldout=set(json.loads((base/'ENVIRONMENT_HOLDOUT_SPLIT.json').read_text())['held_out_environments'])
   for env,details in split['environment_splits'].items():
    for episode in details['train_episode_indices']: split_lookup[(env,int(episode))]='train'
    for record in details['test']: split_lookup[(env,int(record['episode_index']))]='test'
 task_rows=[]
 for directory in sorted(base.glob('*_lerobot')):
  env=directory.name.removesuffix('_lerobot')
  info=json.loads((directory/'meta/info.json').read_text())
  with (directory/'meta/episodes.jsonl').open() as handle:
   metadata=[json.loads(line) for line in handle if line.strip()]
  for ep in metadata:
   idx=int(ep['episode_index']); tag=split_lookup[(env,idx)]
   if ep.get('split') is not None and ep['split'] != tag:
    raise RuntimeError(f'Split conflict: {task} {env} {idx}')
   video=directory/f'videos/chunk-{idx//1000:03d}/observation.images.image/episode_{idx:06d}.mp4'
   if not video.is_file(): raise FileNotFoundError(video)
   row={'task':task,'environment':env,'episode_index':idx,
    'source_episode_index':ep.get('source_episode_index'),
    'split':tag,'domain':'ood' if env in heldout else 'id',
    'training_eligible':tag=='train' and env not in heldout,
    'first_batch_query_eligible':tag=='test' and env not in heldout,
    'level':ep.get('skill_gear',ep.get('level')),
    'length':ep['length'],'fps':info['fps'],'video_path':str(video),
    'skill_annotation':ep.get('skill_annotation'),
    'door_action_selection_eligible':task!='door' or door_minimum[env] is not None}
   task_rows.append(row)
 all_rows.extend(task_rows)
 envstats=[]
 for env in sorted({r['environment'] for r in task_rows}):
  rows=[r for r in task_rows if r['environment']==env]
  stats={'environment':env,'domain':rows[0]['domain'],
   'train':sum(r['training_eligible'] for r in rows),
   'query_test':sum(r['first_batch_query_eligible'] for r in rows),
   'original_train':sum(r['split']=='train' for r in rows),
   'original_test':sum(r['split']=='test' for r in rows),'total':len(rows)}
  stats['level_counts']={str(level):dict(Counter(r['split'] for r in rows if r['level']==level))
   for level in sorted({r['level'] for r in rows if r['level'] is not None})}
  envstats.append(stats)
 summary[task]={'total':len(task_rows),'train':sum(r['training_eligible'] for r in task_rows),
  'id_test':sum(r['first_batch_query_eligible'] for r in task_rows),
  'ood':sum(r['domain']=='ood' for r in task_rows),'environments':envstats}
 for tag in ('train','test'):
  selected=[]
  for env in sorted({r['environment'] for r in task_rows}):
   candidates=[r for r in task_rows if r['environment']==env and r['split']==tag]
   if task=='ball': candidates=[r for r in candidates if r['level']==5] or candidates
   if task=='door':
    desired=door_minimum[env] or 10
    candidates=[r for r in candidates if r['level']==desired] or candidates
   selected.append(random.Random(9709).choice(candidates))
  cells=[]
  for row in selected:
   cap=cv2.VideoCapture(row['video_path'])
   n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   frames=[]
   for frame_idx in (0,n//2,n-1):
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx); ok,frame=cap.read()
    if not ok: raise RuntimeError((row['video_path'],frame_idx))
    scale=400/frame.shape[1]
    frame=cv2.resize(frame,(400,round(frame.shape[0]*scale)))
    cell=np.full((frame.shape[0]+35,400,3),245,np.uint8)
    cell[35:]=frame
    label=f"{row['environment']} {tag} ep{row['episode_index']} L{row['level']} f{frame_idx}"
    cv2.putText(cell,label,(4,22),cv2.FONT_HERSHEY_SIMPLEX,.39,(0,0,0),1,cv2.LINE_AA)
    frames.append(cell)
   cap.release()
   cells.append(np.hstack(frames))
  # Split longer sheets to keep individual objects visible.
  for part in range(0,len(cells),5):
   path=out/'contact_sheets'/f'{task}_{tag}_{part//5}.jpg'
   cv2.imwrite(str(path),np.vstack(cells[part:part+5]),[cv2.IMWRITE_JPEG_QUALITY,93])
 print(task,json.dumps(summary[task],ensure_ascii=False),flush=True)
with (out/'episodes.jsonl').open('w') as handle:
 for row in all_rows: handle.write(json.dumps(row)+'\n')
(out/'dataset_summary.json').write_text(json.dumps(summary,indent=2))
print('AUDIT_ROOT',out)
