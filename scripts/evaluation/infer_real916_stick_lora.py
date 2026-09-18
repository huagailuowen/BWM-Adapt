#!/usr/bin/env python3
"""Stick Standard LoRA TTA with the original frozen support/query protocol."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml

from real916_stick_final_common import ROOT, read_rows, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    cli = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    import torch
    import imageio.v2 as imageio
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one usable GPU required')
    from infer_real97_standard_reference import copy_cached, snapshot
    from infer_real97_reference_trial import labeled, write_video
    config = read(cli.config)
    output = ROOT/config['inference_output']
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'.run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepared = ROOT/config['prepared']
    if not (prepared/'prepared_complete.json').exists():
        raise RuntimeError('Frozen cohort preparation incomplete')
    frozen = output/'input_manifest'
    for name in ('query.jsonl','support.jsonl','plan.json','stage_files.txt','action_stats.json'):
        snapshot(prepared/name, frozen/name)
    for name in ('training_config.yaml','checkpoint_selection.json'):
        snapshot(prepared/'standard'/name, frozen/name)
    snapshot(cli.config, frozen/'evaluation_config.json')
    identity = read(frozen/'checkpoint_selection.json')
    if identity['step'] != config['checkpoint_step']:
        raise RuntimeError('Wrong Standard checkpoint')
    plan = read(frozen/'plan.json')
    queries, supports = [read_rows(frozen/name) for name in ('query.jsonl','support.jsonl')]
    if len(queries)!=63 or len(plan['environments'])!=9 or sum(r['dataset_split']=='test' for r in queries)!=45:
        raise RuntimeError('Changed query population')
    standard = ROOT/config['standard']
    complete = read(standard/'inference_complete.json')
    if complete['model_step']!=identity['step'] or complete['queries']!=len(queries):
        raise RuntimeError('Standard baseline mismatch')
    if (standard/'input_manifest/query.jsonl').read_bytes()!=(frozen/'query.jsonl').read_bytes():
        raise RuntimeError('Standard baseline queries differ')
    for env in plan['environments']:
        chosen = [supports[i] for i in env['support_indices']]
        if len(chosen)!=2 or any(r['dataset_split']!='train' or r['environment']!=env['environment'] for r in chosen):
            raise RuntimeError('Wrong support population')
        if {r['episode_index'] for r in chosen} & {queries[i]['episode_index'] for i in env['query_indices']}:
            raise RuntimeError('Support/query episode leakage')
    flat = yaml.safe_load((frozen/'training_config.yaml').read_text())['training']
    if tuple(flat[k] for k in ('num_frames','num_history_frames','frame_stride','height','width','action_dim','action_type'))!=(33,1,3,192,256,14,'eef_target'):
        raise RuntimeError('Training geometry/action mismatch')
    if flat.get('physical_context_mode')!='none' or flat.get('physical_adapter_mode')!='none':
        raise RuntimeError('LoRA must start from Standard')
    cache = Path('/tmp')/os.environ['USER']/'bwm_shared_cache'
    wan = cache/'Wan2.2-TI2V-5B'
    copy_cached(ROOT/'models/Wan2.2-TI2V-5B',wan,directory=True)
    tag = hashlib.sha256(str(prepared.resolve()).encode()).hexdigest()[:16]
    local_data = cache/'eval_datasets'/('stick916_'+tag)
    copy_cached(Path(plan['source_dataset']),local_data,directory=True,files=prepared/'stage_files.txt')
    checkpoint = cache/'eval_checkpoints'/f'stick916_standard_{tag}.safetensors'
    copy_cached(prepared/'standard/model.safetensors',checkpoint)
    flat.update(dataset_base_path=str(local_data),dataset_metadata_path=str(frozen/'query.jsonl'),
        action_stat_path=str(frozen/'action_stats.json'),model_paths=str(wan),ckpt_path=str(checkpoint),
        output_path=str(output/'raw/lora'),seed=config['seed'],num_inference_steps=config['num_inference_steps'],
        cfg_scale=1.,fps=7,quality=8,max_samples=0,video_light_augmentation_enabled=False,
        spatial_loss_mode='none',use_gradient_checkpointing=True,use_gradient_checkpointing_offload=False)
    runtime = output/'runtime.yaml'
    runtime.write_text(yaml.safe_dump({'inference':flat},sort_keys=False))
    sys.path.insert(0,str(ROOT/'scripts'))
    import infer_stage2_ttt as ttt
    from infer import build_infer_dataset, build_pipeline, prepare_sample_for_rollout, _run_autoregressive
    from wan_video_action.parsers import add_general_config, merge_yaml_and_args
    from wan_video_action.utils import set_global_seed
    from scripts.methods import infer_lora_tta_event80 as lora_engine
    parser = add_general_config(argparse.ArgumentParser())
    if '--frame_stride' not in parser._option_string_actions:
        parser.add_argument('--frame_stride',type=int,default=3)
    args = merge_yaml_and_args(str(runtime),parser,parser.parse_args(['--config',str(runtime)]))
    args.stage2_fixed_timestep_index = None
    lora = config['lora']
    args.lora_steps = lora['steps']
    args.lora_learning_rate = lora['learning_rate']
    args.lora_gradient_clip = lora['gradient_clip_norm']
    original_prepare = lora_engine._prepare_loss_inputs
    def prepare_loss(pipe, data, z, arguments):
        inputs = original_prepare(pipe,data,z,arguments)
        shared = ttt._shared_inputs(inputs)
        shared['_stick_valid_future'] = (int(data['valid_video_frames'])-1)//4
        return inputs
    def masked_loss(pipe, inputs, arguments):
        shared = dict(ttt._shared_inputs(inputs))
        valid = shared.pop('_stick_valid_future')
        if not 1<=valid<=8:
            raise RuntimeError('No complete real future latent block')
        timestep = ttt._sample_timestep(pipe,shared,arguments)
        noise = torch.randn_like(shared['input_latents'])
        shared['latents'] = pipe.scheduler.add_noise(shared['input_latents'],noise,timestep)
        target = pipe.scheduler.training_target(shared['input_latents'],noise,timestep)
        shared['latents'][:,:,:1] = shared['first_frame_latents']
        models = {name:getattr(pipe,name) for name in pipe.in_iteration_models}
        prediction = pipe.model_fn(**models,**shared,timestep=timestep)
        return torch.nn.functional.mse_loss(prediction[:,:,1:1+valid].float(),target[:,:,1:1+valid].float())*pipe.scheduler.training_weight(timestep)
    # Process-local: preserve old Event80 and other-task LoRA implementations.
    lora_engine._prepare_loss_inputs = prepare_loss
    lora_engine._flow_match_loss = masked_loss
    dataset = build_infer_dataset(args)
    support_args = copy.copy(args)
    support_args.dataset_metadata_path = str(frozen/'support.jsonl')
    support_dataset = build_infer_dataset(support_args)
    set_global_seed(config['seed'])
    pipe = build_pipeline(args)
    ttt._freeze_pipe(pipe)
    adapters = lora_engine.install_lora(pipe.dit,targets=set(lora['targets']),rank=lora['rank'],alpha=lora['alpha'],seed=lora['adapter_seed'])
    support_sha = hashlib.sha256((frozen/'support.jsonl').read_bytes()).hexdigest()
    write_json(output/'provenance.json',{'config':config,'checkpoint_identity':identity,
        'support_sha256':support_sha,'support_loss':config['support_loss'],
        'base_model_and_action_encoder_frozen':True,'query_updates_lora':False,
        'adapter_parameter_count':sum(a.lora_a.numel()+a.lora_b.numel() for _,a in adapters),
        'old_outputs_modified':False})
    count = 0
    for env in plan['environments']:
        name = env['environment']
        expected = {'checkpoint_identity':identity,'environment':name,'support_sha256':support_sha,
                    'support_indices':env['support_indices'],'lora':lora}
        adapter_path = output/'adapters'/(name+'.pt')
        if adapter_path.exists():
            state = torch.load(adapter_path,map_location='cpu',weights_only=True)
            if state['expected']!=expected:
                raise RuntimeError('Saved adapter belongs to another protocol')
            with torch.no_grad():
                for key,adapter in adapters:
                    adapter.lora_a.copy_(state['weights'][key]['a'])
                    adapter.lora_b.copy_(state['weights'][key]['b'])
            losses = state['losses']
        else:
            items = []
            for i in env['support_indices']:
                item = support_dataset[i]
                item['valid_video_frames'] = supports[i]['valid_video_frames']
                items.append(item)
            set_global_seed(lora['adapter_seed']+int(env['environment_index'])*1009)
            losses = lora_engine.adapt_lora(pipe,items,adapters,args)
            if len(losses)!=lora['steps'] or not np.isfinite(losses).all():
                raise RuntimeError('Incomplete/nonfinite adaptation')
            weights = {}
            for key,adapter in adapters:
                if not torch.isfinite(adapter.lora_a).all() or not torch.isfinite(adapter.lora_b).all():
                    raise RuntimeError('Nonfinite adapter weights')
                weights[key] = {'a':adapter.lora_a.detach().cpu(),'b':adapter.lora_b.detach().cpu()}
            adapter_path.parent.mkdir(parents=True,exist_ok=True)
            temp = adapter_path.with_suffix('.pt.partial')
            torch.save({'expected':expected,'weights':weights,'losses':losses},temp)
            os.replace(temp,adapter_path)
            del items
        for _,adapter in adapters:
            adapter.lora_a.grad = adapter.lora_b.grad = None
        write_json(output/'adaptation'/(name+'.json'),{**expected,'losses':losses})
        pipe.eval()
        columns = {'train':[],'test':[]}
        for i in env['query_indices']:
            row = queries[i]
            key = f"q{i:04d}_{name}_{row['dataset_split']}_ep{row['episode_index']:06d}"
            baseline = read(standard/'completed'/(key+'.json'))
            if baseline['row']!=row:
                raise RuntimeError('Standard query identity mismatch')
            paths = {k:output/'raw'/k/(key+'.mp4') for k in ('gt','lora')}
            marker = output/'completed'/(key+'.json')
            if not marker.exists():
                sample = dataset[i]
                if tuple(sample['video'].shape)!=(1,3,33,192,256) or tuple(sample['action'].shape)!=(1,33,14):
                    raise RuntimeError('Wrong Stick input tensors')
                for p in paths.values():p.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(baseline['paths']['gt'],paths['gt'])
                args.seed = config['seed']+i
                set_global_seed(args.seed)
                rollout = prepare_sample_for_rollout(copy.copy(sample),i,pipe,args)
                rollout['output_path'] = str(paths['lora'])
                with torch.no_grad():_run_autoregressive(pipe,rollout,args)
                frames = imageio.mimread(str(paths['lora']))
                if len(frames)!=33:raise RuntimeError('Wrong rollout length')
                write_video(paths['lora'],frames,config['fps'])
                write_json(marker,{'index':i,'row':row,'paths':{k:str(p) for k,p in paths.items()},
                    'standard_prediction':baseline['paths']['standard'],'checkpoint_step':identity['step']})
                del sample
                torch.cuda.empty_cache()
            videos = {k:imageio.mimread(str(p)) for k,p in paths.items()}
            videos['standard'] = imageio.mimread(baseline['paths']['standard'])
            if any(len(v)!=33 for v in videos.values()):raise RuntimeError('Comparison length mismatch')
            column = [np.vstack([labeled(np.asarray(videos[k][f]),f"{k} | {row['dataset_split']} ep{row['episode_index']}") for k in ('gt','standard','lora')]) for f in range(33)]
            write_video(output/'comparisons'/(key+'.mp4'),column,config['fps'])
            columns[row['dataset_split']].append(column)
            count += 1
            write_json(output/'progress.json',{'completed_queries':count,'expected_queries':len(queries),'environment':name})
            print(f'[query_done] {key} {count}/{len(queries)}',flush=True)
        for split,values in columns.items():
            if values:
                write_video(output/'grids'/split/(name+'.mp4'),(np.hstack([v[f] for v in values]) for f in range(33)),config['fps'])
    write_json(output/'inference_complete.json',{'method':'lora','model_step':identity['step'],
        'queries':count,'test_queries':45,'environments':9,'query_updates_lora':False,'metric_status':'pending'})


if __name__=='__main__':main()
