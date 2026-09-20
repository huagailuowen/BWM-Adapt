#!/usr/bin/env python3
"""Isolated sim-LR Stick ablation: full-table mean, unchanged support loss."""
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

from real916_stick_final_common import ROOT, plot_pca, read_rows, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--method',choices=['ours','standard'],required=True)
    own=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Compute allocation required')
    import torch
    import imageio.v2 as imageio
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one GPU required')
    config=json.loads(own.config.read_text());setting=config['methods'][own.method]
    prepared=ROOT/config['prepared'];output=ROOT/setting['output'];output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (prepared/'prepared_complete.json').is_file():
        raise RuntimeError('Cohort preparation incomplete')
    from scripts.evaluation.infer_real97_standard_reference import copy_cached, snapshot
    from scripts.evaluation.infer_real97_reference_trial import write_video, labeled
    for name in ['plan.json','query.jsonl','support.jsonl','action_stats.json','stage_files.txt','evaluation_config.json']:
        snapshot(prepared/name,output/'input_manifest'/name)
    reference=prepared/own.method
    for name in ['training_config.yaml','checkpoint_selection.json']:
        snapshot(reference/name,output/'input_manifest'/name)
    plan=json.loads((prepared/'plan.json').read_text());queries=read_rows(prepared/'query.jsonl');supports=read_rows(prepared/'support.jsonl')
    expected_queries=int(plan['query_count'])
    if len(queries)!=expected_queries:
        raise RuntimeError('Frozen query count mismatch')
    query_reference=None;reference_supports=[]
    if config.get('reuse_query_reference_from'):
        query_reference=ROOT/config['reuse_query_reference_from']
        if own.method!='ours':raise RuntimeError('Query reference reuse is Ours-only')
        for filename, expected_file in [('query.jsonl',prepared/'query.jsonl'),
                                        ('context_table.json',reference/'context_table.json'),
                                        ('checkpoint_selection.json',reference/'checkpoint_selection.json')]:
            if (query_reference/'input_manifest'/filename).read_bytes()!=expected_file.read_bytes():
                raise RuntimeError('Reference mismatch: '+filename)
        previous_config=json.loads((query_reference/'provenance.json').read_text())['config']
        if previous_config['seed']!=config['seed'] or previous_config['stage2']!=config['stage2']:
            raise RuntimeError('Reference seed or adaptation schedule changed')
        reference_supports=read_rows(query_reference/'input_manifest/support.jsonl')
    if own.method=='ours' and config.get('reuse_contexts_from'):
        previous=ROOT/config['reuse_contexts_from']
        if (previous/'input_manifest/support.jsonl').read_bytes()!=(prepared/'support.jsonl').read_bytes():
            raise RuntimeError('Cannot reuse adapted Z with different support')
        if (previous/'input_manifest/context_table.json').read_bytes()!=(reference/'context_table.json').read_bytes():
            raise RuntimeError('Cannot reuse adapted Z with different context table')
        if (previous/'input_manifest/checkpoint_selection.json').read_bytes()!=(reference/'checkpoint_selection.json').read_bytes():
            raise RuntimeError('Cannot reuse adapted Z with different checkpoint')
        for env in plan['environments']:
            snapshot(previous/'contexts'/(env['environment']+'.json'),output/'contexts'/(env['environment']+'.json'))
    cache=Path('/tmp')/os.environ['USER']/'bwm_shared_cache'
    wan=cache/'Wan2.2-TI2V-5B';copy_cached(ROOT/'models/Wan2.2-TI2V-5B',wan,directory=True)
    tag=hashlib.sha256(str(prepared.resolve()).encode()).hexdigest()[:16]
    local=cache/'eval_datasets'/('stick916_'+tag)
    copy_cached(Path(plan['source_dataset']),local,directory=True,files=prepared/'stage_files.txt')
    checkpoint_tag=hashlib.sha256(str((ROOT/config.get('checkpoint_cache_prepared',config['prepared'])).resolve()).encode()).hexdigest()[:16]
    checkpoint=cache/'eval_checkpoints'/f'stick916_{own.method}_{checkpoint_tag}.safetensors'
    copy_cached(reference/'model.safetensors',checkpoint)
    flat=yaml.safe_load((reference/'training_config.yaml').read_text())['training']
    expected={'num_frames':33,'num_history_frames':1,'frame_stride':3,'height':192,'width':256,'action_type':'eef_target','action_dim':14}
    if any(flat[k]!=v for k,v in expected.items()):
        raise RuntimeError('Training/inference geometry or action contract mismatch')
    flat.update(dataset_base_path=str(local),dataset_metadata_path=str(prepared/'query.jsonl'),
                action_stat_path=str(prepared/'action_stats.json'),model_paths=str(wan),ckpt_path=str(checkpoint),
                output_path=str(output/'raw'),seed=config['seed'],num_inference_steps=50,cfg_scale=1.,
                video_light_augmentation_enabled=False,spatial_loss_mode='none',fps=7,quality=8,max_samples=0,
                use_gradient_checkpointing=True,use_gradient_checkpointing_offload=False)
    flat.update(config['stage2'])
    runtime=output/'runtime.yaml';runtime.write_text(yaml.safe_dump({'inference':flat},sort_keys=False))
    sys.path.insert(0,str(ROOT/'scripts'))
    import infer_stage2_ttt as ttt
    import infer
    from wan_video_action.utils import save_video,set_global_seed
    sys.argv=['infer_stage2_ttt','--config',str(runtime),'--stage2_ckpt_path',str(checkpoint),'--support_metadata_path',str(prepared/'support.jsonl')]
    args=ttt.parse_args()
    original_prepare=ttt._prepare_loss_inputs
    def prepare_loss(pipe,data,z,arguments):
        inputs=original_prepare(pipe,data,z,arguments)
        shared=inputs[0] if isinstance(inputs,tuple) else inputs
        shared['_stick_valid_future']=(int(data['valid_video_frames'])-1)//4
        return inputs
    def masked_loss(pipe,inputs,arguments):
        shared=dict(ttt._shared_inputs(inputs));valid=int(shared.pop('_stick_valid_future'))
        if not 1<=valid<=8:
            raise ValueError('No complete real future VAE block')
        timestep=ttt._sample_timestep(pipe,shared,arguments)
        noise=torch.randn_like(shared['input_latents'])
        shared['latents']=pipe.scheduler.add_noise(shared['input_latents'],noise,timestep)
        target=pipe.scheduler.training_target(shared['input_latents'],noise,timestep)
        shared['latents'][:,:,:1]=shared['first_frame_latents']
        models={name:getattr(pipe,name) for name in pipe.in_iteration_models}
        predicted=pipe.model_fn(**models,**shared,timestep=timestep)
        loss=torch.nn.functional.mse_loss(predicted[:,:,1:1+valid].float(),target[:,:,1:1+valid].float())
        return loss*pipe.scheduler.training_weight(timestep)
    # Process-local only; legacy entrypoints keep their original loss.
    ttt._prepare_loss_inputs=prepare_loss;ttt._flow_match_loss=masked_loss
    args.stage2_context_clamp_min=-float('inf');args.stage2_context_clamp_max=float('inf')
    dataset=infer.build_infer_dataset(args)
    support_args=copy.copy(args);support_args.dataset_metadata_path=str(prepared/'support.jsonl')
    support_dataset=infer.build_infer_dataset(support_args)
    set_global_seed(config['seed']);pipe=infer.build_pipeline(args);ttt._freeze_pipe(pipe)
    table=None;lookup={};contexts={}
    if own.method=='ours':
        snapshot(reference/'context_table.json',output/'input_manifest/context_table.json')
        table=json.loads((reference/'context_table.json').read_text())
        lookup={int(r['friction_mu']):torch.tensor(r['context'],device=pipe.device,dtype=torch.float32) for r in table['records']}
        plot_pca(table,plan['environments'],output,setting['step'])
    write_json(output/'provenance.json',{'config':config,'method':own.method,'model_step':setting['step'],
               'query_sha256':hashlib.sha256((prepared/'query.jsonl').read_bytes()).hexdigest(),
               'support_sha256':hashlib.sha256((prepared/'support.jsonl').read_bytes()).hexdigest(),
               'support_loss':'training-matched complete real future VAE blocks only',
               'known_environment_initialization':False,'context_hard_bounds':None,
               'inference_light_augmentation':False,'roi':False,'fps':20/3})
    results=[];fps=20/3
    for env in plan['environments']:
        name=env['environment'];state_path=output/'contexts'/(name+'.json')
        unchanged_support=(query_reference is not None and
            all(supports[j]==reference_supports[j] for j in env['support_indices']))
        if unchanged_support and not state_path.exists():
            snapshot(query_reference/'contexts'/(name+'.json'),state_path)
        adapted=None
        if own.method=='ours':
            if state_path.exists():
                state=json.loads(state_path.read_text());adapted=torch.tensor(state['context'],device=pipe.device,dtype=torch.float32)
            else:
                support=[]
                for i in env['support_indices']:
                    if supports[i]['dataset_split']!='train' or supports[i]['environment']!=name:
                        raise RuntimeError('Support leakage')
                    item=support_dataset[i];item['valid_video_frames']=supports[i]['valid_video_frames'];support.append(item)
                target=lookup[env['environment_index']]
                seed=(config['seed']+int(hashlib.sha256(name.encode()).hexdigest()[:8],16))%(2**63-1)
                generator=torch.Generator(device='cpu').manual_seed(seed)
                if config.get('initialization') != 'mean_training_table':
                    raise RuntimeError('This isolated entrypoint requires the full training-table mean')
                initial=torch.stack(list(lookup.values())).mean(0)
                noise=torch.zeros_like(initial)
                set_global_seed(config['seed']+env['environment_index']*1009)
                adapted,losses,_,metrics,trajectory=ttt._adapt_ttt_state(pipe,support,args,adapter_named_params=[],base_adapter_state={},
                    initial_context=initial,target_context=None,
                    trajectory_meta={'environment':name,'sample_index':env['query_indices'][0],'friction_mu':float(env['environment_index'])})
                if not torch.isfinite(adapted).all():raise RuntimeError('Nonfinite adapted Z')
                state={'context':adapted.detach().cpu().tolist(),'initial':initial.cpu().tolist(),'target':target.cpu().tolist(),
                       'noise_seed':None,'noise_range':None,'trajectory':trajectory,'losses':losses,'metrics':metrics,
                       'support_indices':env['support_indices'],'initialization':'mean_training_table','initialization_latent_count':len(lookup)}
                write_json(state_path,state);del support
            contexts[name]=state
        split_columns={'train':[],'test':[]}
        for i in env['query_indices']:
            row=queries[i]
            if row['episode_index'] in {supports[j]['episode_index'] for j in env['support_indices']}:
                raise RuntimeError('Support/query episode overlap')
            key=f"q{i:04d}_{name}_{row['dataset_split']}_ep{row['episode_index']:06d}"
            variants=['stage1','stage2'] if own.method=='ours' else ['standard']
            paths={kind:output/'raw'/kind/(key+'.mp4') for kind in ['gt']+variants}
            marker=output/'completed'/(key+'.json')
            if not marker.exists():
                sample=dataset[i]
                if tuple(sample['video'].shape)!=(1,3,33,192,256) or tuple(sample['action'].shape)!=(1,33,14):
                    raise RuntimeError('Sample shape differs from training')
                paths['gt'].parent.mkdir(parents=True,exist_ok=True)
                reused=set()
                if query_reference is not None:
                    for kind in ['gt','stage1']+(['stage2'] if unchanged_support else []):
                        paths[kind].parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(query_reference/'raw'/kind/(key+'.mp4'),paths[kind])
                        reused.add(kind)
                else:
                    save_video(sample['video'],output_path=str(paths['gt']),fps=7,quality=8)
                for variant in variants:
                    if variant in reused:
                        args.seed=config['seed']+i
                        continue
                    rollout=infer.prepare_sample_for_rollout(copy.copy(sample),i,pipe,args)
                    rollout['output_path']=str(paths[variant]);paths[variant].parent.mkdir(parents=True,exist_ok=True)
                    if own.method=='ours':rollout['physical_context']=lookup[env['environment_index']] if variant=='stage1' else adapted
                    args.seed=config['seed']+i;set_global_seed(args.seed)
                    with torch.no_grad():infer._run_autoregressive(pipe,rollout,args)
                for kind,path in paths.items():
                    if kind not in reused:write_video(path,imageio.mimread(str(path)),fps)
                write_json(marker,{'index':i,'row':row,'paths':{k:str(v) for k,v in paths.items()}})
                del sample;torch.cuda.empty_cache()
            videos={kind:imageio.mimread(str(path)) for kind,path in paths.items()}
            if any(len(frames)!=33 for frames in videos.values()):raise RuntimeError('Frame count mismatch')
            column=[np.vstack([labeled(videos[k][f],f"{k} | {row['dataset_split']} ep{row['episode_index']}") for k in ['gt']+variants]) for f in range(33)]
            write_video(output/'comparisons'/(key+'.mp4'),column,fps)
            split_columns[row['dataset_split']].append(column)
            results.append({'index':i,'environment':name,'split':row['dataset_split'],'paths':{k:str(v) for k,v in paths.items()},
                            'evaluation_frame_indices':row['evaluation_frame_indices']})
            write_json(output/'progress.json',{'completed_queries':len(results),'expected_queries':expected_queries,'results':results})
            print(f'[query_done] {key} {len(results)}/{expected_queries}',flush=True)
        for split,columns in split_columns.items():
            if not columns:continue
            write_video(output/'grids'/split/(name+'.mp4'),[np.hstack([col[f] for col in columns]) for f in range(33)],fps)
        if table is not None:plot_pca(table,plan['environments'],output,setting['step'],contexts)
    write_json(output/'inference_complete.json',{'method':own.method,'model_step':setting['step'],'environments':9,
               'queries':len(results),'test_queries':sum(r['split']=='test' for r in results),'train_queries':sum(r['split']=='train' for r in results),'metric_status':'not_scored',
               'formal_metric_approved':False,'support_selection_review_pending':True})


if __name__=='__main__':
    main()
