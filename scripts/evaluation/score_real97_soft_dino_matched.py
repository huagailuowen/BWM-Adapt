#!/usr/bin/env python3
"""Matched Soft DINO/Standard/Ours scores; no action selection metric."""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for site in (ROOT / ".venv-real97-eval-20260909/lib").glob("python*/site-packages"):
    sys.path.append(str(site))
import cv2
import numpy as np
import score_real97_soft_balanced9_static_matched as base

DINO = ROOT / "outputs/infer_real97_soft_dino_k4_step5500_train117150_static_reference_20260916_v1"
OURS = ROOT / "outputs/infer_real97_soft_balanced9_family_mean_static_20260914_v1"
STANDARD = ROOT / "outputs/infer_real97_soft_standard_matched_balanced9_static_20260914_v1"
OLD = ROOT / "outputs/eval_real97_soft_family_mean_comparison_20260914_v1"
OUT = ROOT / "outputs/eval_real97_soft_dino_matched_20260916_v1"
RESULT = ROOT / "results/real97_soft_dino_matched_20260916_v1"
METHODS = ("standard", "stage1", "ours", "dino")
read, write = base.read, base.write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run scoring on an allocated compute node")
    cv2.setNumThreads(1)
    rows = base.legacy.jsonl(DINO / "query.jsonl")
    reference = base.legacy.jsonl(OURS / "query.jsonl")
    if rows != reference or len(rows) != 36:
        raise RuntimeError("DINO and Ours must use identical 36 factual query rows")
    if sum(r["dataset_split"] == "test" for r in rows) != 18:
        raise RuntimeError("Expected 18 test queries")
    frozen = read(OLD / "shard00/config.json")
    OUT.mkdir(parents=True, exist_ok=True)
    return rows, frozen


def case(index, row):
    key = base.legacy.key_for(index, row)
    marker = read(DINO / "completed" / (key + ".json"))
    if marker["query"] != row or marker["checkpoint_step"] != 5500 or not marker["paired_gt"]:
        raise RuntimeError("DINO completion marker does not match the scoring contract")
    common = copy.deepcopy(row)
    common.update(length=29, native_frame_indices=row["native_frame_indices"][:29],
                  evaluation_frame_indices=[i for i in row["evaluation_frame_indices"] if i <= 28])
    paths = {
        "gt": OURS / "raw/gt" / (key + ".mp4"),
        "standard": STANDARD / "raw/standard" / (key + ".mp4"),
        "stage1": OURS / "raw/stage1" / (key + ".mp4"),
        "ours": OURS / "raw/stage2" / (key + ".mp4"),
        "dino": DINO / "raw/dino" / (key + ".mp4"),
    }
    return key, common, paths


def track_shard(rows, frozen, shard):
    detect = None
    for index, row in enumerate(rows):
        if index % 6 != shard:
            continue
        key, common, paths = case(index, row)
        destination = OUT / "cases" / key
        destination.mkdir(parents=True, exist_ok=True)
        signature = dict(row=common, tracking=frozen["tracking"], filtering=frozen["filtering"],
                         detector=frozen["detector"], source=str(paths["dino"]), source_sha256=sha(paths["dino"]))
        cached_path = destination / "dino_tracks.json"
        if cached_path.exists():
            if read(cached_path)["signature"] != signature:
                raise RuntimeError("Cached DINO detector inputs changed")
            continue
        if detect is None:
            detect = base.legacy.build_detector(frozen["detector"])
        frames = base.decode(paths["dino"])[:29]
        tracks = base.track(frames, common, detect, frozen["tracking"], frozen["filtering"])
        write(cached_path, dict(signature=signature, tracks=tracks))
        # Independent detection on predicted frames; no GT coordinates feed the tracker.
        selected = (0, 5, 9, 12, 18, 21, 25, 28)
        for name, indices in (("tracking_contact_sheet.jpg", selected[:4]),
                              ("tracking_contact_sheet_middle.jpg", selected[4:])):
            panel = np.hstack([base.legacy.annotated(frames[i], tracks[i], "DINO") for i in indices])
            cv2.imwrite(str(destination / name), panel)
        print(json.dumps(dict(query=key, valid=sum(t["measurement_valid"] for t in tracks), total=29)), flush=True)
    write(OUT / f"track_shard{shard:02d}_complete.json", dict(shard=shard, complete=True))


def image_metrics(rows):
    import torch
    import lpips
    if not torch.cuda.is_available():
        raise RuntimeError("LPIPS scoring requires the allocated GPU")
    torch.set_num_threads(4)
    metric = lpips.LPIPS(net="alex", version="0.1", verbose=False).cuda().eval()
    metric.requires_grad_(False)
    for index, row in enumerate(rows):
        key, common, paths = case(index, row)
        destination = OUT / "cases" / key
        destination.mkdir(parents=True, exist_ok=True)
        signature = {kind: sha(path) for kind, path in paths.items()}
        path = destination / "image_metrics.json"
        if path.exists():
            if read(path)["video_sha256"] != signature:
                raise RuntimeError("Image metric source changed")
            continue
        frames = {kind: (base.decode(path)[4:] if kind == "standard" else base.decode(path)[:29])
                  for kind, path in paths.items()}
        # Confirm DINO exported the same initial frame/GT, not a different query.
        dino_gt = base.decode(DINO / "raw/gt" / (key + ".mp4"))[:29]
        discrepancy = max(float(np.abs(a.astype(float)-b.astype(float)).mean())
                          for a,b in zip(frames["gt"], dino_gt))
        if discrepancy > 3:
            raise RuntimeError(f"GT mismatch: {key}, MAE={discrepancy}")
        wanted = common["evaluation_frame_indices"]
        def tensor(video):
            rgb = np.stack([cv2.cvtColor(video[i], cv2.COLOR_BGR2RGB) for i in wanted])
            return torch.from_numpy(rgb.transpose(0,3,1,2).copy()).cuda().float() / 127.5 - 1
        gt = tensor(frames["gt"])
        scores = {}
        for kind in METHODS:
            scores[kind] = base.image_score(frames["gt"], frames[kind], wanted)
            prediction = tensor(frames[kind])
            values = []
            with torch.inference_mode():
                for start in range(0, len(wanted), 8):
                    values.extend(metric(gt[start:start+8], prediction[start:start+8]).flatten().cpu().tolist())
            scores[kind].update(lpips=float(np.mean(values)), lpips_per_frame=values)
        write(path, dict(scores=scores, video_sha256=signature, gt_codec_mae=discrepancy,
                         evaluation_native_frames=[common["native_frame_indices"][i] for i in wanted],
                         lpips_protocol="AlexNet v0.1, RGB [-1,1], 512x256 INTER_LINEAR, no detection mask"))
        print(json.dumps(dict(query=key, image_metrics={k:{x:v[x] for x in ("psnr_db","ssim","lpips")}
                                                        for k,v in scores.items()})), flush=True)
    write(OUT / "image_metrics_complete.json", dict(complete=True, queries=len(rows)))


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def aggregate(rows, frozen):
    results = []
    cohorts = {r["index"]:r for r in read(STANDARD / "matched_cohort.json")["cases"]}
    for index, row in enumerate(rows):
        key, common, paths = case(index, row)
        destination = OUT / "cases" / key
        tracks = {}
        for kind, old_kind in (("gt","gt"),("standard","standard"),("stage1","stage1"),("ours","stage2")):
            cached = read(OLD / "cases" / key / (old_kind + "_tracks.json"))
            sig = cached["signature"]
            if sig["row"] != common or sig["tracking"] != frozen["tracking"] or sig["filtering"] != frozen["filtering"]:
                raise RuntimeError("Reference tracks use a different query/tracker")
            if sha(sig["source"]) != sha(paths[kind]):
                raise RuntimeError("Reference raw video changed since tracking")
            tracks[kind] = cached["tracks"]
        dino = read(destination / "dino_tracks.json")
        if dino["signature"]["source_sha256"] != sha(paths["dino"]):
            raise RuntimeError("DINO video changed after tracking")
        tracks["dino"] = dino["tracks"]
        wanted = common["evaluation_frame_indices"]
        valid = lambda k,i: tracks[k][i]["measurement_valid"] and tracks[k][i]["center"] is not None
        shared = [i for i in wanted if all(valid(k,i) for k in ("gt",)+METHODS)]
        maskrow = copy.deepcopy(common)
        maskrow["evaluation_frame_indices"] = shared
        natural, matched = {}, {}
        for kind in METHODS:
            natural[kind] = base.legacy.score(tracks["gt"], tracks[kind], common, frozen["config"])
            matched[kind] = base.legacy.score(tracks["gt"], tracks[kind], maskrow, frozen["config"])
            if not wanted or wanted[-1] not in shared:
                matched[kind]["center_fde_px"] = None
        images = read(destination / "image_metrics.json")
        result = dict(index=index, key=key, environment=row["environment"], split=row["dataset_split"],
                      cohort=cohorts[index], requested_frames=len(wanted), common_frames=len(shared),
                      natural=natural, common_mask=matched, images=images["scores"],
                      dino_jump_warnings=sum(bool(t.get("jump_warning")) for t in tracks["dino"]),
                      formal_metric_approved=False)
        write(destination / "metrics.json", result)
        results.append(result)
        # Matched overlays enable visual audit without consulting GT to track DINO.
        panel_rows = []
        for kind in ("gt",)+METHODS:
            decoded = base.decode(paths[kind])
            decoded = decoded[4:] if kind == "standard" else decoded[:29]
            panel_rows.append(np.hstack([base.legacy.annotated(decoded[i], tracks[kind][i], kind)
                                        for i in (0,9,18,28)]))
        cv2.imwrite(str(destination / "all_methods_contact_sheet.jpg"), np.vstack(panel_rows))
    groups = {
        "test_all9": [r for r in results if r["split"]=="test"],
        "test_shared6": [r for r in results if r["split"]=="test" and r["cohort"]["standard_environment_seen"]],
        "train_all9": [r for r in results if r["split"]=="train"],
    }
    for env in sorted({r["environment"] for r in results}):
        groups["test/"+env] = [r for r in results if r["split"]=="test" and r["environment"]==env]
    statistics = {}
    for group, items in groups.items():
        total = sum(i["requested_frames"] for i in items)
        statistics[group] = {}
        for kind in METHODS:
            statistics[group][kind] = dict(
                queries=len(items), psnr_db=mean([i["images"][kind]["psnr_db"] for i in items]),
                ssim=mean([i["images"][kind]["ssim"] for i in items]),
                lpips=mean([i["images"][kind]["lpips"] for i in items]),
                common_ADE_px=mean([i["common_mask"][kind]["center_ade_px"] for i in items]),
                natural_ADE_px=mean([i["natural"][kind]["center_ade_px"] for i in items]),
                final_FDE_px=mean([i["natural"][kind]["center_fde_px"] for i in items]),
                common_final_FDE_px=mean([i["common_mask"][kind]["center_fde_px"] for i in items]),
                paired_coverage=sum(i["natural"][kind]["paired_frames"] for i in items)/total,
                common_coverage=sum(i["common_frames"] for i in items)/total,
                final_valid_queries=sum(i["natural"][kind]["center_fde_px"] is not None for i in items))
    summary = dict(complete=True, queries=len(results), groups=statistics, no_action_metric=True,
                   formal_metric_approved=False, manual_review_required=True,
                   image_resolution=[512,256], native_future_frames=list(range(3,85,3)),
                   common_mask_methods=list(METHODS), center="center of visible in-frame box",
                   caveats=["Standard trained only six of these environments and uses five repeated initial frames.",
                            "Ours uses known-family mean initialization; DINO uses four train supports.",
                            "ADE common mask is recomputed including DINO, not the previous global-stage2 mask.",
                            "Missing detections are not zero error; FDE requires actual final eligible frame."])
    write(OUT / "summary.json", summary)
    write(OUT / "per_query.json", results)
    write(RESULT / "summary.json", summary)
    write(RESULT / "per_query.json", results)
    fields = ["cohort","method"] + list(statistics["test_all9"]["standard"])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    report = ["# Soft matched metrics: Standard, Ours and DINO", "",
              "Provisional object tracking; image scores are complete. No action metric.",
              "Common future: native frames 3,6,...,84; no condition frames or padding.", ""]
    for cohort, methods in statistics.items():
        for kind, values in methods.items():
            writer.writerow(dict(cohort=cohort,method=kind,**values))
        if cohort not in ("test_all9","test_shared6","train_all9"):
            continue
        report += ["## "+cohort, "", "| Method | PSNR | SSIM | LPIPS | Common ADE | Final FDE | Pair coverage |",
                   "|---|---:|---:|---:|---:|---:|---:|"]
        for kind,v in methods.items():
            fmt = lambda x: "NA" if x is None else f"{x:.5f}"
            report.append("| "+kind+" | "+" | ".join(fmt(v[k]) for k in
                ("psnr_db","ssim","lpips","common_ADE_px","final_FDE_px","paired_coverage"))+" |")
        report.append("")
    report += ["## Comparability", ""] + ["- "+c for c in summary["caveats"]]
    (RESULT / "metrics.csv").write_text(buffer.getvalue())
    (RESULT / "README.md").write_text("\n".join(report)+"\n")
    write(OUT / "scoring_complete.json", dict(complete=True, formal_metric_approved=False))
    print(json.dumps(statistics["test_all9"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("track","image","aggregate"))
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    rows, frozen = setup()
    if args.mode == "track":
        if not 0 <= args.shard < 6:
            raise ValueError("Invalid shard")
        track_shard(rows, frozen, args.shard)
    elif args.mode == "image":
        image_metrics(rows)
    else:
        aggregate(rows, frozen)

