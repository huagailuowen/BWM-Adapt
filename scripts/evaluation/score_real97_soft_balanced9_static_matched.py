#!/usr/bin/env python3
"""Common-timestamp Soft metrics with independent per-video tracking and audit."""
import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import html
import json
import math
import os
from pathlib import Path
import sys
import time

import score_real97_soft_static_objectcentric_cpu as legacy
import cv2
import numpy as np
from track_real97_soft_dino_cpu import select_detection
from wan_video_action.real97_eval.open_vocab_filter import filter_proposals

ROOT = legacy.ROOT
KINDS = ("gt", "standard", "stage1", "stage2")
read = legacy.read_json
write = legacy.write_json


def decode(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (160, 320, 3):
                raise RuntimeError(f"Unexpected video size: {path}: {frame.shape}")
            frames.append(cv2.resize(frame, (512, 256), interpolation=cv2.INTER_LINEAR))
    finally:
        capture.release()
    if len(frames) != 33:
        raise RuntimeError(f"Expected 33 raw frames, got {len(frames)}: {path}")
    return frames


def track(frames, row, detect, tracking, filtering):
    states, previous = [], None
    for index, frame in enumerate(frames):
        native = int(row["native_frame_indices"][index])
        if states and native == states[-1]["native_frame"]:
            state = copy.deepcopy(states[-1])
            state.update(frame=index, repeated_tail=True)
            states.append(state)
            continue
        start = time.monotonic()
        proposals = filter_proposals(frame, detect(frame), filtering)
        selected, reason = select_detection(proposals, previous, native, tracking)
        state = dict(frame=index, native_frame=native, timestamp_seconds=native / 20,
                     center=None, measurement_valid=False, boundary_partial=False,
                     reason=reason, selected=selected, proposals=proposals,
                     detector_seconds=time.monotonic() - start, jump_warning=False)
        if selected is not None:
            x0, y0, x1, y1 = selected["box"]
            visible = [max(0., x0), max(0., y0), min(512., x1), min(256., y1)]
            if visible[2] > visible[0] and visible[3] > visible[1]:
                center = [(visible[0] + visible[2]) / 2, (visible[1] + visible[3]) / 2]
                displacement = math.dist(center, previous[1]) if previous else None
                margin = tracking["boundary_margin_px"]
                state.update(center=center, visible_box=visible, measurement_valid=True,
                             boundary_partial=bool(min(x0, y0) <= margin or x1 >= 512-margin or y1 >= 256-margin),
                             confidence=float(selected["score"]), displacement_since_last_detection_px=displacement,
                             jump_warning=bool(displacement is not None and displacement > tracking["jump_warning_px"]))
                previous = (native, center)
        states.append(state)
    return states


def image_score(gt, prediction, indices):
    values = []
    for index in indices:
        x, y = gt[index].astype(np.float64), prediction[index].astype(np.float64)
        mse = float(np.mean((x-y)**2))
        ux, uy = cv2.GaussianBlur(x, (11,11), 1.5), cv2.GaussianBlur(y, (11,11), 1.5)
        vx = cv2.GaussianBlur(x*x, (11,11), 1.5)-ux*ux
        vy = cv2.GaussianBlur(y*y, (11,11), 1.5)-uy*uy
        cov = cv2.GaussianBlur(x*y, (11,11), 1.5)-ux*uy
        ssim = ((2*ux*uy+6.5025)*(2*cov+58.5225))/((ux*ux+uy*uy+6.5025)*(vx+vy+58.5225))
        values.append(dict(frame=index, psnr_db=10*math.log10(255**2/max(mse,1e-12)),
                           ssim=float(ssim[5:-5,5:-5].mean())))
    return dict(frames=len(values), psnr_db=float(np.mean([v["psnr_db"] for v in values])) if values else None,
                ssim=float(np.mean([v["ssim"] for v in values])) if values else None,
                protocol="Gaussian-window RGB-channel SSIM; native crop; common unpadded future only")


def trajectory(path, tracks):
    colors = dict(gt="#202020", standard="#d9822b", stage1="#2166ac", stage2="#1b9e77")
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="530">',
           '<rect width="900" height="530" fill="white"/>']
    for i, kind in enumerate(KINDS):
        svg.append(f'<text x="{60+i*200}" y="25" fill="{colors[kind]}" font-family="sans-serif">{kind}</text>')
    for axis, maximum, top in ((0,512,55),(1,256,285)):
        svg.append(f'<rect x="60" y="{top}" width="800" height="185" fill="#fafafa" stroke="#888"/>')
        svg.append(f'<text x="15" y="{top+95}" font-family="sans-serif">{"x" if axis==0 else "y"}</text>')
        for kind in KINDS:
            segment = []
            for state in tracks[kind]+[dict(measurement_valid=False)]:
                if state["measurement_valid"]:
                    segment.append(f'{60+800*state["native_frame"]/84:.2f},{top+185*(1-state["center"][axis]/maximum):.2f}')
                elif segment:
                    svg.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{colors[kind]}" stroke-width="2"/>')
                    segment = []
    svg.append('<text x="60" y="510" font-family="sans-serif">Native frames 0..84; missing measurements are gaps, not interpolation.</text></svg>')
    path.write_text("\n".join(svg))


def process(index, row, config, detect, tracking, filtering, cohort):
    source, standard = ROOT/config["source_inference"], ROOT/config["standard_inference"]
    output = ROOT/config["output"]
    key = legacy.key_for(index, row)
    destination = output/"cases"/key
    destination.mkdir(parents=True, exist_ok=True)
    a, b = read(source/"completed"/(key+".json")), read(standard/"completed"/(key+".json"))
    if a["row"] != row or a["observed_frames"] != 1 or b["observed_frames"] != 5:
        raise RuntimeError(f"Unexpected observation contract: {key}")
    if not a["paired_gt"] or not b["paired_gt"]:
        raise RuntimeError("Unpaired action sweep entered factual evaluation")
    for field in ("environment", "dataset_split", "episode_index", "video", "action", "total_frames"):
        if row[field] != b["row"][field]:
            raise RuntimeError(f"Standard identity mismatch: {key}/{field}")
    if b["row"]["native_frame_indices"][4:] != row["native_frame_indices"][:29]:
        raise RuntimeError("Do not align unlike physical timestamps")
    frames = {kind: decode(a["paths"][kind])[:29] for kind in ("gt", "stage1", "stage2")}
    frames["standard"] = decode(b["paths"]["standard"])[4:]
    standard_gt = decode(b["paths"]["gt"])[4:]
    gt_codec_mae = max(float(np.mean(np.abs(x.astype(float)-y.astype(float)))) for x,y in zip(frames["gt"],standard_gt))
    if gt_codec_mae > 3:
        raise RuntimeError(f"GT sources do not agree within codec tolerance: {key}: {gt_codec_mae}")
    common_row = copy.deepcopy(row)
    common_row.update(length=29, native_frame_indices=row["native_frame_indices"][:29],
                      evaluation_frame_indices=[i for i in row["evaluation_frame_indices"] if i <= 28])
    tracks = {}
    for kind in KINDS:
        path = destination/(kind+"_tracks.json")
        signature = dict(row=common_row, method=kind, tracking=tracking, filtering=filtering,
                         source=a["paths"].get(kind,b["paths"].get(kind)))
        if path.is_file():
            cached = read(path)
            if cached["signature"] != signature:
                raise RuntimeError("Cached tracks use a different query or detector configuration")
            tracks[kind] = cached["tracks"]
        else:
            tracks[kind] = track(frames[kind], common_row, detect, tracking, filtering)
            write(path, dict(signature=signature, tracks=tracks[kind]))
        print(f"[track] {key} {kind} {sum(s['measurement_valid'] for s in tracks[kind])}/29", flush=True)
    common_indices = [i for i in common_row["evaluation_frame_indices"]
                      if all(tracks[k][i]["measurement_valid"] for k in KINDS)]
    metrics, common_metrics, images = {}, {}, {}
    masked = copy.deepcopy(common_row)
    masked["evaluation_frame_indices"] = common_indices
    for kind in KINDS[1:]:
        metrics[kind] = legacy.score(tracks["gt"], tracks[kind], common_row, config)
        common_metrics[kind] = legacy.score(tracks["gt"], tracks[kind], masked, config)
        # FDE always means the true last eligible future, never last detected frame.
        common_metrics[kind]["center_fde_px"] = metrics[kind]["center_fde_px"] if common_row["evaluation_frame_indices"] and common_row["evaluation_frame_indices"][-1] in common_indices else None
        images[kind] = image_score(frames["gt"], frames[kind], common_row["evaluation_frame_indices"])
    from infer_real97_reference_trial import write_video, labeled
    comparison = [np.vstack([labeled(cv2.resize(cv2.cvtColor(frames[k][i],cv2.COLOR_BGR2RGB),(320,160)),
                                f'{k} | {row["dataset_split"]} ep{row["episode_index"]} f{common_row["native_frame_indices"][i]}')
                             for k in KINDS]) for i in range(29)]
    write_video(output/"comparisons"/(key+".mp4"), comparison, config["playback_fps"])
    panels = [np.hstack([legacy.annotated(frames[k][i],tracks[k][i],k) for i in (0,9,18,28)]) for k in KINDS]
    cv2.imwrite(str(destination/"contact_sheet.jpg"), np.vstack(panels))
    middle = [np.hstack([legacy.annotated(frames[k][i],tracks[k][i],k) for i in (5,12,21,25)]) for k in KINDS]
    cv2.imwrite(str(destination/"contact_sheet_middle.jpg"), np.vstack(middle))
    write_video(destination/"tracking_overlay.mp4",
                (cv2.cvtColor(np.vstack([legacy.annotated(frames[k][i],tracks[k][i],k) for k in KINDS]),cv2.COLOR_BGR2RGB)
                 for i in range(29)), config["playback_fps"])
    trajectory(destination/"trajectory.svg", tracks)
    result = dict(index=index,key=key,environment=row["environment"],split=row["dataset_split"],row=row,
                  cohort=cohort,requested_frames=len(common_row["evaluation_frame_indices"]),
                  all_methods_common_frames=len(common_indices),metrics=metrics,common_mask_metrics=common_metrics,
                  image_metrics=images,gt_codec_mae_max=gt_codec_mae,
                  track_diagnostics={k:dict(reasons=dict(Counter(s["reason"] for s in t)),
                                            jump_warnings=sum(s.get("jump_warning",False) for s in t),
                                            boundary_frames=sum(s.get("boundary_partial",False) for s in t))
                                     for k,t in tracks.items()},formal_metric_approved=False)
    write(destination/"summary.json", result)
    print("[query_done] "+json.dumps({"key":key,"ADE":{k:v["center_ade_px"] for k,v in metrics.items()},
                                      "coverage":{k:v["coverage"] for k,v in metrics.items()}}),flush=True)


def aggregate(rows, config):
    output = ROOT/config["output"]
    summaries = [read(output/"cases"/legacy.key_for(i,r)/"summary.json") for i,r in enumerate(rows)]
    groups = defaultdict(list)
    for item in summaries:
        groups["all"].append(item)
        groups["split/"+item["split"]].append(item)
        cohort = "both_env_seen" if item["cohort"]["standard_environment_seen"] else "ours_ID_standard_OOD"
        groups[cohort+"/"+item["split"]].append(item)
        groups["environment/"+item["environment"]+"/"+item["split"]].append(item)
    mean = lambda values: float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None
    statistics = {}
    for group, items in groups.items():
        methods = {}
        for kind in KINDS[1:]:
            metrics = [i["metrics"][kind] for i in items]
            common = [i["common_mask_metrics"][kind] for i in items]
            wanted = sum(i["requested_frames"] for i in items)
            methods[kind] = dict(
                queries=len(items),episode_mean_ADE_px=mean([m["center_ade_px"] for m in metrics]),
                mean_FDE_px=mean([m["center_fde_px"] for m in metrics]),
                paired_coverage=sum(m["paired_frames"] for m in metrics)/wanted if wanted else None,
                gt_coverage=sum(m["gt_valid_frames"] for m in metrics)/wanted if wanted else None,
                prediction_coverage=sum(m["prediction_valid_frames"] for m in metrics)/wanted if wanted else None,
                complete_horizon_queries=sum(m["complete_horizon"] for m in metrics),
                final_valid_queries=sum(m["center_fde_px"] is not None for m in metrics),
                all_methods_common_mask_ADE_px=mean([m["center_ade_px"] for m in common]),
                common_mask_coverage=sum(i["all_methods_common_frames"] for i in items)/wanted if wanted else None,
                static_baseline_ADE_px=mean([m["static_baseline_ade_on_paired_frames_px"] for m in metrics]),
                psnr_db=mean([i["image_metrics"][kind]["psnr_db"] for i in items]),
                ssim=mean([i["image_metrics"][kind]["ssim"] for i in items]))
        statistics[group] = methods
    write(output/"summary_provisional.json",dict(expected_queries=len(rows),completed_queries=len(summaries),
         groups=statistics,formal_metric_approved=False,manual_review_required=True,
         no_action_selection_metric=True,center_definition="center of visible in-frame detection box",
         common_horizon="native 3,6,...,84, excluding duplicate tail and observed frame0",
         fairness="Standard has five identical frame0 observations and Ours one; training environment sets differ; report shared and OOD cohorts separately.",
         caution="Coverage is not accuracy. Missing centers stay missing, never silently counted as zero error."))
    write(output/"per_query.json",summaries)
    review = [dict(key=i["key"],split=i["split"],environment=i["environment"],
                   contact_sheet=str(output/"cases"/i["key"]/"contact_sheet.jpg"),
                   middle_sheet=str(output/"cases"/i["key"]/"contact_sheet_middle.jpg"),
                   diagnostics=i["track_diagnostics"]) for i in summaries]
    write(output/"review_manifest.json",review)
    import imageio.v2 as imageio
    from infer_real97_reference_trial import write_video
    for env in sorted({r["environment"] for r in rows}):
        selected = [s for s in summaries if s["environment"]==env]
        for split in ("train","test","all"):
            cases = [s for s in selected if split=="all" or s["split"]==split]
            videos = [imageio.mimread(str(output/"comparisons"/(s["key"]+".mp4"))) for s in cases]
            if videos:
                write_video(output/"grids"/f"{env}_gt_standard_stage1_stage2_{split}.mp4",
                            (np.hstack([video[i] for video in videos]) for i in range(29)),config["playback_fps"])
    page = ['<!doctype html><html><meta charset="utf-8"><body><h1>Soft matched static tracking audit</h1>',
            '<p>Rows: GT / Standard / Stage1 / Stage2. Common 28 future frames. Green boxes and red crosses are provisional tracker measurements.</p>']
    for item in summaries:
        key = html.escape(item["key"])
        page += [f'<h2>{key}</h2><a href="cases/{key}/tracking_overlay.mp4">Tracking overlay</a>',
                 f'<a href="cases/{key}/trajectory.svg">Trajectory</a>',
                 f'<p><img width="1536" loading="lazy" src="cases/{key}/contact_sheet.jpg"></p>']
    (output/"index.html").write_text("\n".join(page+["</body></html>"]))
    write(output/"scoring_complete.json",dict(queries=len(summaries),formal_metric_approved=False))
    print("[aggregate] "+json.dumps(statistics.get("split/test",{})),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--aggregate-only",action="store_true")
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Video processing must run on a compute node")
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    cv2.setNumThreads(1)
    config=read(args.config)
    output=ROOT/config["output"]
    output.mkdir(parents=True,exist_ok=True)
    rows=legacy.jsonl(ROOT/config["source_inference"]/"query.jsonl")
    if len(rows)!=config["expected_queries"] or any(r["observed_native_frame_indices"] != [0] for r in rows):
        raise RuntimeError("Unexpected latest Soft query population")
    if args.aggregate_only:
        aggregate(rows,config)
        return
    if not 0<=args.shard<config["shards"]:
        raise RuntimeError("Invalid shard")
    shard=output/f"shard{args.shard:02d}"
    shard.mkdir(exist_ok=True)
    lock=(shard/".writer.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tracking=read(ROOT/config["tracking_config"])
    filtering=read(ROOT/config["filter_config"])
    detector=read(ROOT/config["detector_config"])
    frozen=dict(config=config,tracking=tracking,filtering=filtering,detector=detector)
    frozen_path=shard/"config.json"
    if frozen_path.is_file() and read(frozen_path)!=frozen:
        raise RuntimeError("Refusing to mix metric versions")
    write(frozen_path,frozen)
    cohort={r["index"]:r for r in read(ROOT/config["standard_inference"]/"matched_cohort.json")["cases"]}
    pending=[(i,r) for i,r in enumerate(rows) if i%config["shards"]==args.shard
             and not (output/"cases"/legacy.key_for(i,r)/"summary.json").is_file()]
    if not pending:
        return
    detect=legacy.build_detector(detector)
    failures=[]
    for index,row in pending:
        try:
            process(index,row,config,detect,tracking,filtering,cohort[index])
        except Exception as error:
            import traceback
            traceback.print_exc()
            failures.append(dict(index=index,error=repr(error)))
        write(shard/"progress.json",dict(last_index=index,failures=failures))
    if failures:
        raise RuntimeError(f"Failed queries preserved for review: {failures}")
    write(shard/"complete.json",dict(done=True))


if __name__=="__main__":
    main()
