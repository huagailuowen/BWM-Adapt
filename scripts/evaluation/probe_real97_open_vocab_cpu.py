#!/usr/bin/env python3
"""Open-vocabulary detector probe on CPU; never loads the BWM world model."""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# Reuse installed torch/transformers read-only; cv2/numpy come from the eval venv.
sys.path.append(str(ROOT/".venv/lib/python3.10/site-packages"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU Slurm allocation required; no login-node model loading")
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoConfig,AutoProcessor,AutoModelForZeroShotObjectDetection

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK","4")))
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    cfg=json.loads(args.config.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"config_snapshot.json").write_text(json.dumps(cfg,indent=2))
    cache=ROOT/cfg["cache_dir"]
    cache.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    print("[detector] downloading/loading "+cfg["model_id"],flush=True)
    processor=AutoProcessor.from_pretrained(cfg["model_id"],cache_dir=str(cache))
    model_config=AutoConfig.from_pretrained(cfg["model_id"],cache_dir=str(cache))
    model_config.disable_custom_kernels=True
    model=AutoModelForZeroShotObjectDetection.from_pretrained(
        cfg["model_id"],config=model_config,cache_dir=str(cache),
        use_safetensors=True,trust_remote_code=False).to("cpu").eval()
    load_seconds=time.monotonic()-started
    print(f"[detector] CPU model ready in {load_seconds:.1f}s",flush=True)
    inventory=[json.loads(s) for s in (ROOT/cfg["inventory"]).read_text().splitlines() if s.strip()]
    lookup={(r["environment"],r["episode_index"]):r for r in inventory if r["task"]=="soft"}
    labels=json.loads((ROOT/cfg["landmark_config"]).read_text())["validation_landmarks"]
    samples=[]
    for label in labels:
        row=lookup[(label["environment"],label["episode_index"])]
        samples.append(dict(row,frame=label["frame"],reference_center=label["center"],
                            kind="coarse_landmark"))
    for env,ep in cfg["failure_episodes"]:
        row=lookup[(env,ep)]
        for f in (0,row["length"]//2):
            samples.append(dict(row,frame=f,kind="previous_whole_episode_miss"))
    for i in range(6):
        path=ROOT/cfg["negative_root"]/f"negative_{i:02d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Required negative control missing: {path}")
        samples.append(dict(image_path=str(path),kind="synthetic_negative",frame=0,
                            environment="negative",episode_index=i))
    (args.output/"sample_manifest.json").write_text(json.dumps(samples,indent=2))
    records=[]
    signature=inspect.signature(processor.post_process_grounded_object_detection).parameters
    threshold_name="threshold" if "threshold" in signature else "box_threshold"
    with (args.output/"detections.jsonl").open("w") as handle:
        for idx,sample in enumerate(samples):
            if "image_path" in sample:
                image=cv2.imread(sample["image_path"])
            else:
                cap=cv2.VideoCapture(sample["video_path"])
                cap.set(cv2.CAP_PROP_POS_FRAMES,sample["frame"])
                ok,image=cap.read()
                cap.release()
                if not ok:
                    raise RuntimeError(f"Cannot decode {sample}")
            if image is None:
                raise RuntimeError(f"Cannot read {sample}")
            start=time.monotonic()
            pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            inputs=processor(images=pil,text=cfg["prompt"],return_tensors="pt",
                             size=cfg["detector_resize"])
            with torch.inference_mode():
                outputs=model(**inputs)
            kwargs={threshold_name:cfg["box_threshold"],"text_threshold":cfg["text_threshold"],
                    "target_sizes":[(image.shape[0],image.shape[1])]}
            detected=processor.post_process_grounded_object_detection(
                outputs,inputs.input_ids,**kwargs)[0]
            detections=[]
            names=detected.get("text_labels",detected.get("labels",[]))
            for i,(box,score) in enumerate(zip(detected["boxes"],detected["scores"])):
                x0,y0,x1,y1=map(float,box.tolist())
                width,height=x1-x0,y1-y0
                plausible=(cfg["min_box_side"]<=width<=cfg["max_box_side"] and
                           cfg["min_box_side"]<=height<=cfg["max_box_side"])
                detections.append(dict(box=[x0,y0,x1,y1],
                                       center=[(x0+x1)/2,(y0+y1)/2],
                                       score=float(score),
                                       label=str(names[i]) if i<len(names) else "",
                                       plausible_terminal_size=plausible))
            detections.sort(key=lambda r:r["score"],reverse=True)
            # Select by detector score/geometry only. Never choose nearest to GT.
            selected=next((r for r in detections if r["plausible_terminal_size"]),None)
            reference=sample.get("reference_center")
            error=None if selected is None or reference is None else float(np.linalg.norm(
                np.array(selected["center"])-np.array(reference)))
            row=dict(sample,all_detections=detections,selected=selected,
                     center_error_px=error,seconds=time.monotonic()-start,
                     formally_certified=False)
            for box in detections[:8]:
                x0,y0,x1,y1=map(round,box["box"])
                color=(0,255,0) if box is selected else (0,160,255)
                cv2.rectangle(image,(x0,y0),(x1,y1),color,1)
                cv2.putText(image,f'{box["score"]:.2f}',(x0,max(12,y0-3)),0,.4,color,1)
            if reference is not None:
                cv2.drawMarker(image,tuple(map(round,reference)),(255,0,255),cv2.MARKER_CROSS,10,1)
            cv2.putText(image,f'{sample["environment"]} ep{sample["episode_index"]} f{sample["frame"]}',
                        (4,17),0,.4,(0,0,255),1)
            path=args.output/f"sample{idx:03d}.jpg"
            cv2.imwrite(str(path),image)
            row["review_path"]=str(path)
            handle.write(json.dumps(row)+"\n");handle.flush()
            records.append(row)
            print("[sample] "+json.dumps(dict(index=idx,kind=sample["kind"],
                    environment=sample["environment"],episode=sample["episode_index"],
                    frame=sample["frame"],selected=selected,center_error_px=error,
                    seconds=row["seconds"])),flush=True)
    landmarks=[r for r in records if r["kind"]=="coarse_landmark"]
    negatives=[r for r in records if r["kind"]=="synthetic_negative"]
    errors=[r["center_error_px"] for r in landmarks if r["center_error_px"] is not None]
    summary=dict(model=cfg["model_id"],load_seconds=load_seconds,samples=len(records),
                 landmarks=len(landmarks),landmarks_detected=len(errors),
                 landmarks_within6=sum(e<=6 for e in errors),
                 landmark_mean_error_px=float(np.mean(errors)) if errors else None,
                 negative_controls=len(negatives),
                 negative_false_detections=sum(r["selected"] is not None for r in negatives),
                 mean_seconds_per_image=float(np.mean([r["seconds"] for r in records])),
                 formally_certified=False)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print("[summary] "+json.dumps(summary),flush=True)


if __name__=="__main__":
    main()

