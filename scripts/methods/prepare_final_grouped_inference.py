"""Prepare a resumable formal evaluation from a completed training allocation."""
import argparse
import json
import os
import re
import struct
from pathlib import Path
import yaml

p=argparse.ArgumentParser()
p.add_argument('--kind',choices=['event80','multibackground'],required=True)
p.add_argument('--job',type=int,required=True)
p.add_argument('--dim',type=int,required=True)
a=p.parse_args()
if a.kind=='event80':
    directory=Path(f'outputs/method_benchmarks/pushbox_friction_event80/ours_context_dim_{a.dim}/seed_20260708_job_{a.job}/checkpoints')
    training=Path(f'tmp/run_configs/event80_active35_c{a.dim}_iterative_{a.job}.yaml')
    log=Path(f'logs/methods/event80-c{a.dim}-iter-{a.job}.out')
    bench=Path('results/pushbox_friction_event80/event80_grid_id5_ood5_k1_oracle_informative_support25_60_v1')
    manifest=bench/'protocol/support_query_manifest.json'
    order=[0,20,40,59,79,30,10,69,49,35,25,54,15,64,5,74,44,38,42,33,46,28,51,23,56,18,61,13,66,8,71,3,76,39,41]
    metric=yaml.safe_load(Path('configs/evaluation/event80/informative_support_complete_benchmark.yaml').read_text())
    seed=20260708
    slug=f'ours_context_dim_{a.dim}'
else:
    directory=Path('outputs/pushbox_matchedphysics30bg_active20/ours_c32_roi10x_mainview_8env6action_replacement_newlr003_2gpu/121249')
    training=Path('configs/train/train_push_box_matchedphysics30bg_active20_ours_c32_roi10x_mainview_8env6action_replacement_newlr003_2gpu_24h.yaml')
    log=Path(f'logs/methods/pb30a20-o8x6r-{a.job}.out')
    bench=Path('results/pushbox_multibackground/matchedphysics30bg_active20_id5_ood5_k1_oracle_informative_support25_60_v1')
    manifest=bench/'protocol/support_query_manifest.json'
    order=json.loads((directory/'curriculum_group_order.json').read_text())['group_order']
    metric=yaml.safe_load((bench/'methods/ours/step_4400/seed_20260903/benchmark_config.yaml').read_text())
    seed=20260903
    slug='ours_mainview_8env6action'
# Validate the safetensors byte length and its paired JSON without loading weights.
selected=None
for ckpt in sorted(directory.glob('step-*.safetensors'),key=lambda x:int(x.stem.split('-')[1]),reverse=True):
    table=ckpt.with_suffix('.context_table.json')
    if not table.is_file():continue
    try:
        with ckpt.open('rb') as handle:
            length=struct.unpack('<Q',handle.read(8))[0]
            if length>100_000_000:continue
            header=json.loads(handle.read(length))
        end=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
        if ckpt.stat().st_size != 8+length+end:continue
        data=json.loads(table.read_text())
        if not data['records']:continue
    except (ValueError,KeyError,OSError,struct.error):continue
    selected=(ckpt,table,data);break
if selected is None:raise RuntimeError(f'No complete checkpoint/context-table pair in {directory}')
ckpt,table,data=selected
step=int(ckpt.stem.split('-')[1])
phases=re.findall(r'\[curriculum_phase\].*?step=(\d+).*?active_count=(\d+)',log.read_text())
counts=[int(n) for s,n in phases if int(s)<=step]
if not counts:raise RuntimeError('Cannot determine active environment count at checkpoint')
count=counts[-1]
config=yaml.safe_load(training.read_text())
metadata=Path(config['dataset']['dataset_metadata_path'])
rows=[json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
if a.kind=='event80':
    # Event80 identifies friction environments by mu_index, not environment_id.
    by_id={}
    for r in rows:
        if 'mu_index' not in r:
            raise ValueError('Event80 metadata requires mu_index for active-environment mapping')
        index=int(r['mu_index'])
        mu=float(r['friction_mu'])
        if index in by_id and abs(by_id[index]-mu)>1e-8:
            raise ValueError(f'Conflicting friction values for mu_index={index}')
        by_id[index]=mu
    missing=set(order[:count])-set(by_id)
    if missing:
        raise ValueError(f'Active mu_index values missing from metadata: {sorted(missing)}')
    mus=[by_id[i] for i in order[:count]]
else:
    mus=sorted({float(r['friction_mu']) for r in rows})
    mus=[mus[i] for i in order[:count]]
if count>len(order) or len(mus)!=count:
    raise ValueError(f'Invalid active-environment count {count} for curriculum order')
if int(config['physical_context']['physical_context_dim'])!=a.dim:
    raise ValueError('Requested context dimension differs from training configuration')
active=[r for r in data['records'] if any(abs(float(r['friction_mu'])-mu)<1e-8 for mu in mus)]
if len(active)!=count:raise RuntimeError(f'Active table mismatch: {len(active)} != {count}')
out=bench/'methods'/slug/f'train_job_{a.job}'/f'step_{step}'/f'seed_{seed}'
out.mkdir(parents=True,exist_ok=True)
filtered={**data,'num_groups':count,'records':active}
if 'context_shape' in filtered:filtered['context_shape']=[count,*filtered['context_shape'][1:]]
active_path=out/'active_context_table.json'
active_path.write_text(json.dumps(filtered,indent=2)+'\n')
protocol=json.loads(manifest.read_text())
if isinstance(protocol,dict):protocol=protocol['environments']
for r in protocol:
    r.setdefault('source_index',r['support_indices'][0]);r.setdefault('target_indices',r['query_indices'])
local_manifest=out/'input_support_query_manifest.json'
local_manifest.write_text(json.dumps(protocol,indent=2)+'\n')
metric['methods']={slug:str(out.relative_to(bench))}
metric['output_root']=str(bench/'metrics'/f'{slug}_job{a.job}_step{step}')
prepared_metric=out/'prepared_benchmark_config.yaml'
prepared_metric.write_text(yaml.safe_dump(metric,sort_keys=False))
# Reuse the established Ours rollout/grid pipeline with isolated output/cache keys.
s=Path('jobs/run_eval_push_box_matchedphysics30bg_ours4400_low.sh').read_text()
values={'CONFIG':str(training),'CKPT_SOURCE':str(ckpt),'TABLE_SOURCE':str(active_path),'DATASET':metric['dataset_root'],'BENCH':str(bench),'META':metric['metadata_path'],'MANIFEST':str(local_manifest),'OUT':str(out),'ACTIVE':','.join(str(x) for x in mus),'LOCAL_CKPT':f'$CACHE/trained_ckpts/{slug}-job{a.job}-step{step}.safetensors','LOCAL_TABLE':f'$CACHE/trained_ckpts/{slug}-job{a.job}-step{step}.context_table.json'}
for key,value in values.items():s=re.sub(r'^'+key+r'=.*$',lambda m:f'{key}="{value}"',s,flags=re.M)
s=s.replace('pb30a20-ours.lock',f'{slug}-job{a.job}-step{step}.lock')
s=s.replace("protocol[key] for key in ('domain','support_indices','query_indices','target_indices','target_sample_ids','support_selection')","protocol[key] for key in ('domain','support_indices','query_indices','target_indices','target_sample_ids','support_selection') if key in protocol")
s=s.replace('--seed 20260903',f'--seed {seed}')
if a.kind=='event80':s=s.replace('--fps 10','--fps 20')
s=s.replace('.venv/bin/python scripts/evaluation/evaluate_event80_benchmark.py --config "$OUT/benchmark_config.yaml"',f'cp "{prepared_metric}" "$OUT/benchmark_config.yaml"\n.venv/bin/python scripts/evaluation/evaluate_event80_benchmark.py --config "$OUT/benchmark_config.yaml"')
run=out/'run_inference.sh';run.write_text(s)
(out/'checkpoint_selection.json').write_text(json.dumps({'training_job':a.job,'checkpoint':str(ckpt),'context_table':str(table),'step':step,'active_count':count,'active_values':mus,'dimension':a.dim},indent=2)+'\n')
print(f'[selected] checkpoint={ckpt} active={count} output={out}',flush=True)
os.execv('/bin/bash',['bash',str(run)])
