#!/usr/bin/env python3
"""Exact training-loader factual Stage1-table versus support-only Stage2 rollout."""
from __future__ import annotations
import argparse
import colorsys
import copy
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import yaml
from real97_trial_common import ROOT, read_jsonl, require_compute, write_json, write_jsonl


def stage(prepared, output):
    plan=json.loads((prepared/"plan.json").read_text())
    cache=Path("/tmp")/os.environ["USER"]/"bwm_shared_cache"
    cache.mkdir(parents=True,exist_ok=True)
    def copy_locked(source,dest,directory=False,files=None):
        dest.parent.mkdir(parents=True,exist_ok=True)
        marker=Path(str(dest)+".copy_complete")
        with Path(str(dest)+".lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            # Reuse the established Wan cache marker as well.
            if marker.exists() or (directory and (dest/".copy_complete").exists()):
                return
            if directory:
                dest.mkdir(exist_ok=True)
                cmd=["rsync","-a"]+(["--files-from="+str(files)] if files else [])+[str(source)+"/",str(dest)+"/"]
                subprocess.run(cmd,check=True)
            else:
                temporary=Path(str(dest)+".partial")
                subprocess.run(["rsync","-a",str(source),str(temporary)],check=True)
                os.replace(temporary,dest)
            marker.write_text("complete\n")
    model_dir=cache/"Wan2.2-TI2V-5B"
    copy_locked(ROOT/"models/Wan2.2-TI2V-5B",model_dir,True)
    tag=hashlib.sha256(str(prepared.resolve()).encode()).hexdigest()[:16]
    checkpoint=cache/"eval_checkpoints"/(tag+".safetensors")
    copy_locked(prepared/"reference/model.safetensors",checkpoint)
    local_data=cache/"eval_datasets"/tag
    copy_locked(Path(plan["source_dataset"]),local_data,True,prepared/"stage_files.txt")
    source=yaml.safe_load((prepared/"reference/training_config.yaml").read_text())
    flat={k:v for section in source.values() if isinstance(section,dict) for k,v in section.items()}
    flat.update(plan["protocol"]["ttt"])
    flat.update(dataset_base_path=str(local_data),dataset_metadata_path=str(prepared/"query.jsonl"),
                action_stat_path=str(prepared/"reference/action_stats.json"),ckpt_path=str(checkpoint),
                model_paths=str(model_dir),output_path=str(output/"raw/stage2"),
                seed=plan["protocol"]["seed"],max_samples=0,fps=7,quality=8,
                video_light_augmentation_enabled=False,use_gradient_checkpointing=True,
                use_gradient_checkpointing_offload=False,stage2_group_keys="environment")
    runtime=output/"runtime.yaml"
    runtime.write_text(yaml.safe_dump({"inference":flat},sort_keys=False))
    write_json(output/"provenance.json",dict(prepared=str(prepared),plan=plan,checkpoint=str(checkpoint),
                model_source=str(prepared/"reference/model.safetensors"),frames_per_second=20/3))
    return plan,runtime,checkpoint


def write_video(path,frames,fps):
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name("."+path.stem+".partial.mp4")
    with imageio.get_writer(str(temporary),fps=float(fps),codec="libx264",quality=8,macro_block_size=1) as writer:
        for frame in frames: writer.append_data(frame)
    os.replace(temporary,path)


def labeled(frame,label):
    from PIL import Image,ImageDraw
    canvas=Image.new("RGB",(frame.shape[1],frame.shape[0]+22),"white")
    canvas.paste(Image.fromarray(frame),(0,22))
    ImageDraw.Draw(canvas).text((4,4),label,fill="black")
    return np.asarray(canvas)


def pca_plot(path,table,contexts,environments):
    values=np.asarray([row["context"] for row in table["records"]],dtype=float).reshape(len(table["records"]),-1)
    mean=values.mean(0); _,singular,axes=np.linalg.svd(values-mean,full_matrices=False)
    projection=(values-mean)@axes[:2].T
    endpoints=np.stack([(np.asarray(contexts[env["environment"]]["context"]).reshape(-1)-mean)@axes[:2].T for env in environments])
    allxy=np.vstack([projection,endpoints]);lo=allxy.min(0);span=np.maximum(allxy.max(0)-lo,1e-6)
    def xy(p):return 70+690*(p[0]-lo[0])/span[0],540-460*(p[1]-lo[1])/span[1]
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="620"><rect width="1100" height="620" fill="white"/>',
           '<text x="50" y="30" font-family="sans-serif" font-size="18">Training-time Z (circles) and inference-time Z (triangles)</text>']
    lookup={float(row["friction_mu"]):i for i,row in enumerate(table["records"])}
    for i,env in enumerate(environments):
        color="#"+"".join(f"{int(v*255):02x}" for v in colorsys.hsv_to_rgb(i/max(1,len(environments)),.7,.8))
        x,y=xy(projection[lookup[float(env["environment_index"])]])
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}"/>')
        x,y=xy(endpoints[i]);parts.append(f'<polygon points="{x},{y-7} {x-6},{y+5} {x+6},{y+5}" fill="{color}" stroke="black"/>')
        parts.append(f'<text x="800" y="{70+i*26}" fill="{color}" font-family="sans-serif" font-size="12">{html.escape(env["environment"])}</text>')
    total=max(float(np.sum(singular**2)),1e-12)
    parts.append(f'<text x="80" y="590" font-family="sans-serif" font-size="13">PC1 {100*singular[0]**2/total:.1f}%; PC2 {100*singular[1]**2/total:.1f}%; PCA fitted on training table only.</text></svg>')
    path.write_text("\n".join(parts))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepared",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    own=parser.parse_args();require_compute()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Exactly one usable allocated GPU is required; no CPU fallback")
    own.prepared=own.prepared.resolve();own.output=own.output.resolve();own.output.mkdir(parents=True,exist_ok=True)
    plan,runtime,checkpoint=stage(own.prepared,own.output)
    sys.path.insert(0,str(ROOT/"scripts"))
    import infer_stage2_ttt as ttt
    from infer import build_infer_dataset,build_pipeline,prepare_sample_for_rollout,_run_autoregressive
    from wan_video_action.utils import save_video,set_global_seed
    import imageio.v2 as imageio
    sys.argv=["infer_stage2_ttt","--config",str(runtime),"--stage2_ckpt_path",str(checkpoint),
              "--support_metadata_path",str(own.prepared/"support.jsonl")]
    args=ttt.parse_args();set_global_seed(int(args.seed))
    query_rows=read_jsonl(own.prepared/"query.jsonl");support_rows=read_jsonl(own.prepared/"support.jsonl")
    dataset=build_infer_dataset(args)
    support_args=copy.copy(args);support_args.dataset_metadata_path=str(own.prepared/"support.jsonl")
    support_dataset=build_infer_dataset(support_args)
    pipe=build_pipeline(args);ttt._freeze_pipe(pipe)
    table=json.loads((own.prepared/"reference/context_table.json").read_text())
    lookup={float(row["friction_mu"]):torch.tensor(row["context"],device=pipe.device,dtype=torch.float32) for row in table["records"]}
    initial=torch.stack(list(lookup.values())).mean(0)
    contexts={};results=[];fps=20/3
    for env in plan["environments"]:
        name=env["environment"];context_path=own.output/"contexts"/(name+".json")
        if context_path.is_file():
            state=json.loads(context_path.read_text());adapted=torch.tensor(state["context"],device=pipe.device,dtype=torch.float32)
        else:
            support=[support_dataset[index] for index in env["support_indices"]]
            if any(support_rows[index]["dataset_split"]!="train" or support_rows[index]["environment"]!=name for index in env["support_indices"]):
                raise RuntimeError("Support split/environment leakage")
            set_global_seed(int(args.seed)+int(env["environment_index"])*1009)
            adapted,losses,_,metrics,trajectory=ttt._adapt_ttt_state(pipe,support,args,
                adapter_named_params=[],base_adapter_state={},initial_context=initial,target_context=None,
                trajectory_meta=dict(environment=name,sample_index=env["query_indices"][0],friction_mu=float(env["environment_index"])))
            state=dict(context=adapted.detach().cpu().tolist(),initial=initial.cpu().tolist(),losses=losses,
                       trajectory=trajectory,metrics=metrics,support_indices=env["support_indices"])
            write_json(context_path,state)
            del support
        contexts[name]=state
        columns=[]
        for index in env["query_indices"]:
            row=query_rows[index]
            if row["episode_index"] in {support_rows[i]["episode_index"] for i in env["support_indices"]}:
                raise RuntimeError("Query episode overlaps support")
            key=f'q{index:04d}_{name}_{row["dataset_split"]}_ep{row["episode_index"]:06d}'
            paths={kind:own.output/"raw"/kind/(key+".mp4") for kind in ["gt","stage1","stage2"]}
            marker=own.output/"completed"/(key+".json")
            if not marker.is_file():
                sample=dataset[index]
                if tuple(sample["video"].shape)!=(1,3,args.num_frames,args.height,args.width) or tuple(np.shape(sample["action"]))!=(1,args.num_frames,14):
                    raise RuntimeError(f"Training/inference shape mismatch: {sample['video'].shape}, {np.shape(sample['action'])}")
                paths["gt"].parent.mkdir(parents=True,exist_ok=True)
                save_video(sample["video"],output_path=str(paths["gt"]),fps=7,quality=8)
                for kind,z in [("stage1",lookup[float(env["environment_index"])]),("stage2",adapted)]:
                    rollout=prepare_sample_for_rollout(copy.copy(sample),index,pipe,args)
                    rollout.update(physical_context=z,output_path=str(paths[kind]))
                    set_global_seed(int(args.seed)+index)
                    original_seed=args.seed;args.seed=int(plan["protocol"]["seed"])+index
                    _run_autoregressive(pipe,rollout,args);args.seed=original_seed
                # Preserve the exact native 20 Hz / stride3 playback rate in every output.
                for path in paths.values():
                    frames=imageio.mimread(str(path));write_video(path,frames,fps)
                write_json(marker,dict(index=index,row=row,paths={k:str(v) for k,v in paths.items()},paired_gt=True))
                del sample
                torch.cuda.empty_cache()
            videos={kind:imageio.mimread(str(path)) for kind,path in paths.items()}
            if any(len(frames)!=row["length"] for frames in videos.values()):
                raise RuntimeError("GT and predictions differ in frame count")
            column=[np.vstack([labeled(videos[kind][f],f'{kind} | {row["dataset_split"]} ep{row["episode_index"]}')
                               for kind in ["gt","stage1","stage2"]]) for f in range(row["length"])]
            write_video(own.output/"comparisons"/(key+".mp4"),column,fps)
            columns.append(column);results.append(dict(index=index,environment=name,split=row["dataset_split"],paths={k:str(v) for k,v in paths.items()}))
            print(f"[query_done] {key}",flush=True)
        write_video(own.output/"grids"/(name+"_gt_stage1_stage2_train_test.mp4"),
                    [np.hstack([column[f] for column in columns]) for f in range(len(columns[0]))],fps)
        pca_plot(own.output/"training_inference_Z_pca.svg",table,contexts,[e for e in plan["environments"] if e["environment"] in contexts])
        write_json(own.output/"progress.json",dict(completed_environments=list(contexts),query_results=results))
    write_json(own.output/"inference_complete.json",dict(environments=len(contexts),queries=len(results),
                unavailable_environments=plan["unavailable_environments"],metric_status="CPU_scoring_pending"))
    print("[inference_complete] "+str(own.output),flush=True)


if __name__=="__main__":main()
