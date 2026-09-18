#!/usr/bin/env python3
"""Independent Standard / LoRA static-one-frame Soft evaluation on frozen cases."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np
import yaml

from real97_trial_common import ROOT, read_jsonl, write_json
from infer_real97_reference_trial import labeled, write_video
from infer_real97_soft_ours_history5_static import cached_copy, frozen_copy, rgb_video


def read(path):
    return json.loads(Path(path).read_text())


def header_metadata(path):
    # Read the safetensors header only, never load weights on a CPU/login node.
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if size <= 0 or size > 64*1024*1024:
            raise RuntimeError('Invalid safetensors header')
        header = json.loads(f.read(size))
    offsets = [v['data_offsets'][1] for k,v in header.items() if k != '__metadata__']
    if not offsets or path.stat().st_size != 8+size+max(offsets):
        raise RuntimeError('Truncated checkpoint')
    return header.get('__metadata__', {})


def pin_checkpoint(config):
    run = ROOT / config['training_run']
    dest = ROOT / config['snapshot']
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / '.publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (dest / 'complete.json').exists():
            identity = read(dest / 'complete.json')
            if identity['training_job'] != config['training_job']:
                raise RuntimeError('Pinned checkpoint from different training')
            if header_metadata(dest / 'model.safetensors') != identity['metadata']:
                raise RuntimeError('Pinned checkpoint metadata changed')
            return dest, identity
        candidates = []
        for path in (run / 'protected').glob('step-*.safetensors'):
            step = int(path.stem.removeprefix('step-'))
            if step < config['minimum_checkpoint_step']:
                continue
            meta = header_metadata(path)
            if meta.get('method') == 'standard_pooled' and meta.get('reason') in config['allowed_checkpoint_reasons']:
                if int(meta['step']) != step:
                    raise RuntimeError('Checkpoint filename/metadata mismatch')
                candidates.append((step, path, meta))
        if not candidates:
            raise RuntimeError('No complete final/deadline checkpoint; refusing periodic/2300 fallback')
        step, source, meta = max(candidates, key=lambda x:x[0])
        model = dest / 'model.safetensors'
        if model.exists():
            if not os.path.samefile(source, model):
                raise RuntimeError('Conflicting protected model')
        else:
            os.link(source, model)
        frozen_copy(run / 'submitted_config.yaml', dest / 'training_config.yaml')
        frozen_copy(run / 'input_manifest/action_stats.json', dest / 'action_stats.json')
        identity = {'training_job':config['training_job'], 'step':step, 'metadata':meta,
                    'source':str(source), 'bytes':model.stat().st_size,
                    'mtime_ns':model.stat().st_mtime_ns, 'protected_by_hardlink':True}
        write_json(dest / 'complete.json', identity)
        return dest, identity


def prepare(config, output):
    snapshot, identity = pin_checkpoint(config)
    reference = ROOT / config['reference_inference']
    if not (reference / 'inference_complete.json').is_file():
        raise RuntimeError('Frozen Ours inference incomplete')
    for name in ('plan.json','query.jsonl','support.jsonl'):
        frozen_copy(reference / name, output / name)
    for name in ('training_config.yaml','action_stats.json','complete.json'):
        frozen_copy(snapshot / name, output / 'input_manifest' / name)
    plan = read(output / 'plan.json')
    queries, supports = [read_jsonl(output / name) for name in ('query.jsonl','support.jsonl')]
    if len(plan['environments']) != config['expected_environments'] or len(queries) != config['expected_queries']:
        raise RuntimeError('Wrong frozen population')
    if sum(r['dataset_split']=='test' for r in queries) != config['expected_test_queries']:
        raise RuntimeError('Wrong held-out query count')
    flat = yaml.safe_load((snapshot / 'training_config.yaml').read_text())['training']
    if [flat[k] for k in ('num_frames','height','width')] != config['geometry']:
        raise RuntimeError('Training geometry mismatch')
    if (flat['num_history_frames'],flat['frame_stride'],flat['action_type'],flat['action_dim']) != (1,3,'eef_target',14):
        raise RuntimeError('Training history/action contract mismatch')
    if not flat.get('standard_eef_canonicalize') or flat.get('physical_context_mode') != 'none' or flat.get('physical_adapter_mode') != 'none':
        raise RuntimeError('Expected canonical-EEF Standard with no environment Z/adapter')
    source = Path(flat['dataset_base_path'])
    train = read_jsonl(ROOT / config['training_run'] / 'input_manifest/train.jsonl')
    legal = {(r['environment'],r['episode_index'],r['start_frame']) for r in train if r['dataset_split']=='train'}
    train_episodes = {(r['environment'],r['episode_index']) for r in train}
    test_keys = {(r['environment'],r['episode_index']) for r in queries if r['dataset_split']=='test'}
    if test_keys & train_episodes:
        raise RuntimeError('Held-out query leakage into training')
    for env in plan['environments']:
        chosen = [supports[i] for i in env['support_indices']]
        if len(chosen) != config['support_count_per_environment']:
            raise RuntimeError('Support count changed')
        if {r['support_direction'] for r in chosen} != {'left','right'}:
            raise RuntimeError('Supports lack both motion directions')
        for row in chosen:
            if row['dataset_split']!='train' or row['environment']!=env['environment'] or (row['environment'],row['episode_index'],row['start_frame']) not in legal:
                raise RuntimeError('Support not in frozen training legal starts')
        if {r['episode_index'] for r in chosen} & {queries[i]['episode_index'] for i in env['query_indices']}:
            raise RuntimeError('Support/query overlap')
    for row in queries+supports:
        start, last = int(row['start_frame']), int(row['total_frames'])-1
        expected = [min(start+3*i,last) for i in range(33)]
        if row['native_frame_indices'] != expected or row['observed_native_frame_indices'] != [start]:
            raise RuntimeError('Unexpected frame/action timing')
        if row['action_semantics']!='eef_target' or row['length']!=33 or row['frame_stride']!=3:
            raise RuntimeError('Wrong action/video row contract')
        if row in queries and start != 0:
            raise RuntimeError('Query must begin at static episode start')
    files = sorted({p for r in queries+supports for p in r['video']+[r['action'],r['action'].split('/')[0]+'/meta/info.json']})
    if any(Path(p).is_absolute() or '..' in Path(p).parts for p in files):
        raise RuntimeError('Unsafe relative staging path')
    filelist = output / 'input_manifest/stage_files.txt'
    filelist.write_text('\n'.join(files)+'\n')
    cache = Path('/tmp') / os.environ['USER'] / 'bwm_shared_cache'
    wan = cache / 'Wan2.2-TI2V-5B'
    cached_copy(ROOT / 'models/Wan2.2-TI2V-5B', wan, directory=True)
    tag = hashlib.sha256((str(source)+'\n'+'\n'.join(files)).encode()).hexdigest()[:16]
    local_data = cache / 'eval_datasets' / ('soft9_std_lora_'+tag)
    cached_copy(source, local_data, directory=True, files=filelist)
    model_tag = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    model = cache / 'eval_checkpoints' / ('soft9_std_'+model_tag+'.safetensors')
    cached_copy(snapshot / 'model.safetensors', model)
    flat.update(dataset_base_path=str(local_data), dataset_metadata_path=str(output/'query.jsonl'),
                action_stat_path=str(output/'input_manifest/action_stats.json'), model_paths=str(wan),
                ckpt_path=str(model), output_path=str(output/'raw'), seed=config['seed'],
                num_inference_steps=config['num_inference_steps'], cfg_scale=1., fps=7, quality=8,
                video_light_augmentation_enabled=False, spatial_loss_mode='none', max_samples=0,
                use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False)
    runtime = output / 'runtime.yaml'
    runtime.write_text(yaml.safe_dump({'inference':flat},sort_keys=False))
    return plan, queries, supports, runtime, identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--method',choices=['standard','lora'],required=True)
    own = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    import torch
    import imageio.v2 as imageio
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one GPU required')
    config = read(own.config)
    output = ROOT / config['outputs'][own.method]
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/'.run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen_copy(own.config,output/'input_manifest/experiment.json')
    plan, queries, supports, runtime, identity = prepare(config,output)
    sys.path.insert(0,str(ROOT/'scripts'))
    from infer import build_infer_dataset, build_pipeline, prepare_sample_for_rollout, _run_autoregressive
    from infer_stage2_ttt import _freeze_pipe
    from wan_video_action.parsers import add_general_config, merge_yaml_and_args
    from wan_video_action.data.history_prefix import CanonicalTargetEEF
    from wan_video_action.utils import set_global_seed
    parser = add_general_config(argparse.ArgumentParser())
    if '--frame_stride' not in parser._option_string_actions:
        parser.add_argument('--frame_stride',type=int,default=3)
    args = parser.parse_args(['--config',str(runtime)])
    args = merge_yaml_and_args(str(runtime),parser,args)
    args.stage2_fixed_timestep_index = None
    stats = read(output/'input_manifest/action_stats.json')['eef_target']
    def dataset_for(name):
        local = copy.copy(args)
        local.dataset_metadata_path = str(output/name)
        dataset = build_infer_dataset(local)
        dataset.special_operator_map['action'] = CanonicalTargetEEF(local,stats)
        return dataset
    dataset = dataset_for('query.jsonl')
    standard = ROOT / config['outputs']['standard']
    adapters = []
    if own.method=='lora':
        complete = read(standard/'inference_complete.json')
        if complete['queries']!=len(queries) or complete['checkpoint_identity']!=identity:
            raise RuntimeError('Standard comparison has different model/population')
        for name in ('query.jsonl','support.jsonl'):
            if (standard/name).read_bytes()!=(output/name).read_bytes():
                raise RuntimeError('Changed Standard comparison manifests')
        from scripts.methods.infer_lora_tta_event80 import adapt_lora, install_lora
        support_dataset = dataset_for('support.jsonl')
    set_global_seed(config['seed'])
    pipe = build_pipeline(args)
    _freeze_pipe(pipe)
    if own.method=='lora':
        lora = config['lora']
        args.lora_steps = lora['steps']
        args.lora_learning_rate = lora['learning_rate']
        args.lora_gradient_clip = lora['gradient_clip_norm']
        adapters = install_lora(pipe.dit,targets=set(lora['target_modules']),rank=lora['rank'],
                                alpha=lora['alpha'],seed=lora['adapter_seed'])
    support_sha = hashlib.sha256((output/'support.jsonl').read_bytes()).hexdigest()
    write_json(output/'provenance.json',{'config':config,'method':own.method,'checkpoint':identity,
        'condition_frames':1,'predicted_frames':32,'frame_stride':3,'action':'training CanonicalTargetEEF',
        'support_sha256':support_sha,'support_loss':'existing single-frame flow matching, no ROI',
        'query_updates_model':False,'query_updates_lora':False,'source_outputs_modified':False})
    completed = 0
    for env in plan['environments']:
        name = env['environment']
        if own.method=='lora':
            adapter_path = output/'adapters'/(name+'.pt')
            expected = {'identity':identity,'environment':name,'support_sha256':support_sha,
                        'support_indices':env['support_indices'],'lora':config['lora']}
            if adapter_path.exists():
                state = torch.load(adapter_path,map_location='cpu',weights_only=True)
                if state['expected']!=expected:
                    raise RuntimeError('Saved LoRA state mismatch')
                with torch.no_grad():
                    for module_name, adapter in adapters:
                        adapter.lora_a.copy_(state['weights'][module_name]['a'])
                        adapter.lora_b.copy_(state['weights'][module_name]['b'])
                losses = state['losses']
            else:
                set_global_seed(config['lora']['adapter_seed']+int(env['environment_index'])*1009)
                items = [support_dataset[i] for i in env['support_indices']]
                losses = adapt_lora(pipe,items,adapters,args)
                if len(losses)!=config['lora']['steps'] or not np.isfinite(losses).all():
                    raise RuntimeError('LoRA support loss nonfinite/incomplete')
                weights = {}
                for module_name, adapter in adapters:
                    if not torch.isfinite(adapter.lora_a).all() or not torch.isfinite(adapter.lora_b).all():
                        raise RuntimeError('Nonfinite LoRA weights')
                    weights[module_name] = {'a':adapter.lora_a.detach().cpu(),'b':adapter.lora_b.detach().cpu()}
                adapter_path.parent.mkdir(parents=True,exist_ok=True)
                temp = adapter_path.with_suffix('.pt.partial')
                torch.save({'expected':expected,'weights':weights,'losses':losses},temp)
                os.replace(temp,adapter_path)
                del items
            for _, adapter in adapters:
                adapter.lora_a.grad = adapter.lora_b.grad = None
            write_json(output/'adaptation'/(name+'.json'),{**expected,'losses':losses})
        pipe.eval()
        split_columns = {'train':[],'test':[]}
        for index in env['query_indices']:
            row = queries[index]
            key = f"q{index:04d}_{name}_{row['dataset_split']}_ep{row['episode_index']:06d}"
            marker = output/'completed'/(key+'.json')
            paths = {kind:output/'raw'/kind/(key+'.mp4') for kind in ('gt',own.method)}
            if not marker.exists():
                sample = dataset[index]
                if tuple(sample['video'].shape)!=(1,3,33,160,320) or tuple(sample['action'].shape)!=(1,33,14):
                    raise RuntimeError('Action/video tensor mismatch')
                write_video(paths['gt'],rgb_video(sample['video']),config['fps'])
                args.seed = config['seed']+index
                set_global_seed(args.seed)
                rollout = prepare_sample_for_rollout(copy.copy(sample),index,pipe,args)
                paths[own.method].parent.mkdir(parents=True,exist_ok=True)
                rollout['output_path'] = str(paths[own.method])
                with torch.no_grad():
                    _run_autoregressive(pipe,rollout,args)
                frames = imageio.mimread(str(paths[own.method]))
                if len(frames)!=33:
                    raise RuntimeError('Wrong generated frame count')
                write_video(paths[own.method],frames,config['fps'])
                write_json(marker,{'index':index,'row':row,'paths':{k:str(p) for k,p in paths.items()},
                    'prediction':str(paths[own.method]),'query':row,'observed_frames':1,
                    'checkpoint_step':identity['step'],'paired_gt':True})
                del sample
                torch.cuda.empty_cache()
            videos = {k:imageio.mimread(str(p)) for k,p in paths.items()}
            kinds = ['gt',own.method]
            if own.method=='lora':
                baseline = read(standard/'completed'/(key+'.json'))
                if baseline['row']!=row:
                    raise RuntimeError('Standard/LoRA query mismatch')
                videos['standard'] = imageio.mimread(baseline['paths']['standard'])
                kinds = ['gt','standard','lora']
            if any(len(v)!=33 for v in videos.values()):
                raise RuntimeError('Comparison video length mismatch')
            column = [np.vstack([labeled(np.asarray(videos[k][f]),f"{k} | {row['dataset_split']} ep{row['episode_index']}") for k in kinds]) for f in range(33)]
            write_video(output/'comparisons'/(key+'.mp4'),column,config['fps'])
            split_columns[row['dataset_split']].append(column)
            completed += 1
            write_json(output/'progress.json',{'completed_queries':completed,'expected_queries':len(queries),'environment':name})
            print(f'[query_done] {key} {completed}/{len(queries)}',flush=True)
        for split,columns in split_columns.items():
            if columns:
                write_video(output/'grids'/split/(name+'.mp4'),
                    (np.hstack([column[f] for column in columns]) for f in range(33)),config['fps'])
    write_json(output/'inference_complete.json',{'method':own.method,'checkpoint_identity':identity,
        'model_step':identity['step'],'queries':completed,'test_queries':config['expected_test_queries'],
        'environments':len(plan['environments']),'observed_frames':1,'prediction_frames':32,
        'query_updates_lora':False,'source_outputs_modified':False,'metric_status':'not_scored'})


if __name__=='__main__':
    main()
