#!/usr/bin/env python3
"""Read-only trajectory audit; write reports, never alter datasets or training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


TASKS = {
    "ball": ("ball_friction_9_7_crop", 1),
    "door": ("door_close_9_7", 1),
    "stick": ("stick_balance_9_7", 3),
    "soft": ("soft_pull_9_7_crop_512x256", 3),
}
RAW = "raw.target_action.gello_state_raw"
CAL = "raw.target_action.gello_state_calibrated"
CMD = "raw.commanded_target.joint_rad"
CANDIDATE = "raw.target_action.mapped_candidate_joint_rad"
SENT = "raw.commanded_target.arm_sent"
FRESH = "raw.target_action.fresh"
FIELDS = ["action", "observation.eef_state", RAW, CAL, CMD, CANDIDATE, SENT, FRESH]


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "min": float(values.min()),
            "p50": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "p99": float(np.quantile(values, .99)), "max": float(values.max())}


def pair_examples(q, p, starts, ends, dot, angle, displacement, mask, sent, limit=4):
    result = []
    for index in np.flatnonzero(mask)[:limit]:
        a, b = int(starts[index]), int(ends[index])
        result.append({
            "frames": [a, b], "time_s_at_20Hz": [a / 20, b / 20],
            "quaternion_dot": float(dot[index]), "physical_angle_deg": float(angle[index]),
            "position_displacement_mm": float(displacement[index]),
            "previous_q_wxyz": q[a].tolist(), "next_q_wxyz": q[b].tolist(),
            "previous_xyz": p[a].tolist(), "next_xyz": p[b].tolist(),
            "arm_sent": [bool(sent[a]), bool(sent[b])] if sent is not None else None,
        })
    return result


def pose_audit(poses, stride, sent):
    poses = np.asarray(poses, dtype=np.float64)
    p, q = poses[:, :3], poses[:, 3:7].copy()
    norms = np.linalg.norm(q, axis=1)
    valid = np.isfinite(q).all(axis=1) & (norms > 1e-10)
    q[valid] /= norms[valid, None]
    q[~valid] = np.nan
    dominant = np.argmax(np.abs(np.nan_to_num(q)), axis=1)
    summary = {
        "frames": len(poses), "invalid_pose_rows": int((~np.isfinite(poses).all(axis=1)).sum()),
        "invalid_quaternions": int((~valid).sum()),
        "dominant_component_counts": dict(Counter(int(x) for x in dominant[valid])),
        "dominant_component_negative_frames": int(np.count_nonzero(q[np.arange(len(q)), dominant] < 0)),
        "quaternion_component_min": np.nanmin(q, axis=0).tolist(),
        "quaternion_component_max": np.nanmax(q, axis=0).tolist(),
    }
    metrics = {"quaternion_norm_error": np.abs(norms[valid] - 1)}
    first = q[0].copy() if valid[0] else None
    for gap in sorted({1, stride}):
        # Every native start is considered, not just the stride-zero offset.
        starts = np.arange(max(0, len(q) - gap))
        ends = starts + gap
        ok = valid[starts] & valid[ends]
        starts, ends = starts[ok], ends[ok]
        dot = np.clip(np.sum(q[starts] * q[ends], axis=1), -1, 1)
        angle = np.rad2deg(2 * np.arccos(np.abs(dot)))
        displacement = np.linalg.norm(p[ends] - p[starts], axis=1) * 1000
        raw_jump = np.linalg.norm(q[ends] - q[starts], axis=1)
        negative = dot < 0
        large = (angle > 30) | (displacement > 100)
        item = {
            "pairs": len(starts), "negative_quaternion_dot_pairs": int(negative.sum()),
            "antipodal_pairs_physical_angle_under_1deg": int(np.count_nonzero(negative & (angle < 1))),
            "antipodal_pairs_physical_angle_under_10deg": int(np.count_nonzero(negative & (angle < 10))),
            "physical_angle_over_30deg_pairs": int(np.count_nonzero(angle > 30)),
            "position_over_100mm_pairs": int(np.count_nonzero(displacement > 100)),
            "sign_change_examples": pair_examples(q, p, starts, ends, dot, angle, displacement, negative, sent),
            "large_motion_examples": pair_examples(q, p, starts, ends, dot, angle, displacement, large, sent),
        }
        if sent is not None:
            active = sent[starts] & sent[ends]
            item["both_arm_sent_pairs"] = int(active.sum())
            item["negative_dot_both_arm_sent_pairs"] = int(np.count_nonzero(negative & active))
            item["large_motion_both_arm_sent_pairs"] = int(np.count_nonzero(large & active))
        summary[f"gap_{gap}"] = item
        metrics[f"gap_{gap}.physical_angle_deg"] = angle
        metrics[f"gap_{gap}.position_displacement_mm"] = displacement
        metrics[f"gap_{gap}.raw_quaternion_l2_jump"] = raw_jump
    return summary, metrics, first


def joint_audit(joints, sent=None):
    joints = np.asarray(joints, dtype=np.float64)
    valid = np.isfinite(joints).all(axis=1)
    result = {"frames": len(joints), "invalid_rows": int((~valid).sum())}
    result["per_joint_min_rad"] = np.nanmin(joints, axis=0).tolist()
    result["per_joint_max_rad"] = np.nanmax(joints, axis=0).tolist()
    masks = {"all_finite": valid}
    if sent is not None:
        masks["sent_commands"] = valid & sent
        result["arm_sent_true_rows"] = int(sent.sum())
        result["arm_sent_true_invalid_rows"] = int(np.count_nonzero(sent & ~valid))
    metrics = {}
    for name, mask in masks.items():
        indices = np.flatnonzero(mask)
        delta = np.diff(joints[indices], axis=0)
        per_pair = np.max(np.abs(delta), axis=1) if len(delta) else np.array([])
        examples = []
        for index in np.argsort(per_pair)[-3:][::-1]:
            examples.append({"frames": [int(indices[index]), int(indices[index + 1])],
                             "max_joint_change_rad": float(per_pair[index]),
                             "delta_rad": delta[index].tolist()})
        result[name] = {"rows": int(mask.sum()), "pairs": len(delta),
                        "max_joint_delta_over_pi_pairs": int(np.count_nonzero(per_pair > np.pi)),
                        "max_joint_delta_over_0_25rad_pairs": int(np.count_nonzero(per_pair > .25)),
                        "largest_change_examples": examples}
        metrics[f"{name}.max_joint_delta_rad"] = per_pair
        metrics[f"{name}.gap_in_native_frames"] = np.diff(indices)
    return result, metrics


def cross_episode_signs(initials):
    if not initials:
        return {}
    labels = [item[0] for item in initials]
    q = np.stack([item[1] for item in initials])
    dot = np.clip(q @ q.T, -1, 1)
    upper = np.triu(np.ones_like(dot, dtype=bool), 1)
    close = (np.abs(dot) > np.cos(np.deg2rad(5) / 2)) & upper
    opposite = close & (dot < 0)
    indices = np.argwhere(opposite)
    same_env = sum(labels[a].split('/')[0] == labels[b].split('/')[0] for a, b in indices)
    return {
        "episodes": len(labels), "near_orientation_pairs_under_5deg": int(close.sum()),
        "near_orientation_opposite_sign_pairs": int(opposite.sum()),
        "same_environment_opposite_sign_pairs": int(same_env),
        "examples": [{"episodes": [labels[a], labels[b]], "dot": float(dot[a, b]),
                      "first_q": q[a].tolist(), "second_q": q[b].tolist()}
                     for a, b in indices[:5]],
    }


def audit(root, task, stride):
    rows = []
    accumulated = defaultdict(list)
    initials = defaultdict(list)
    counts = Counter()
    field_episodes = Counter()
    flag_counts = Counter()
    env_counts = Counter()
    examples = defaultdict(list)
    for path in sorted(root.glob('*_lerobot/data/*/episode_*.parquet')):
        env = path.parents[2].name
        episode = int(path.stem.removeprefix('episode_'))
        identity = f'{env}/episode_{episode:06d}'
        pf = pq.ParquetFile(path)
        names = set(pf.schema_arrow.names)
        table = pf.read(columns=[name for name in FIELDS if name in names])
        arrays = {k: np.asarray(v.to_pylist(), dtype=np.float64)
                  for k, v in zip(table.column_names, table.columns)}
        n = len(table)
        counts['episodes'] += 1
        counts['frames'] += n
        env_counts[env] += 1
        sent = arrays[SENT].reshape(-1).astype(bool) if SENT in arrays else None
        if sent is not None:
            flag_counts['arm_sent_true'] += int(sent.sum())
            flag_counts['arm_sent_false'] += int((~sent).sum())
        if FRESH in arrays:
            fresh = arrays[FRESH].reshape(-1).astype(bool)
            flag_counts['target_fresh_true'] += int(fresh.sum())
            flag_counts['target_fresh_false'] += int((~fresh).sum())
        row = {'environment': env, 'episode_index': episode, 'num_frames': n, 'fields': {}}
        poses = [name for name in [RAW, CAL, 'observation.eef_state'] if name in arrays]
        if task in ('stick', 'soft'):
            poses.insert(0, 'action')
        for name in poses:
            field_episodes[name] += 1
            info, metrics, first = pose_audit(arrays[name], stride, sent)
            row['fields'][name] = info
            for key, values in metrics.items():
                accumulated[f'{name}|{key}'].append(values)
                info[key] = distribution(values)
            if first is not None:
                initials[name].append((identity, first))
            for key, value in info.items():
                if isinstance(value, int):
                    counts[f'{name}|{key}'] += value
                elif key.startswith('gap_') and '.' not in key and isinstance(value, dict):
                    for metric, number in value.items():
                        if isinstance(number, int):
                            counts[f'{name}|{key}.{metric}'] += number
                    for kind in ('sign_change_examples', 'large_motion_examples'):
                        if len(examples[f'{name}|{key}.{kind}']) < 10:
                            examples[f'{name}|{key}.{kind}'].extend(
                                {'episode': identity, **sample} for sample in value[kind])
        for name in (CMD, CANDIDATE):
            if name not in arrays:
                continue
            field_episodes[name] += 1
            info, metrics = joint_audit(arrays[name], sent)
            row['fields'][name] = info
            for key, values in metrics.items():
                accumulated[f'{name}|{key}'].append(values)
                info[key] = distribution(values)
            for key in ('invalid_rows', 'arm_sent_true_invalid_rows'):
                counts[f'{name}|{key}'] += info.get(key, 0)
            for phase in ('all_finite', 'sent_commands'):
                if phase in info:
                    for key in ('max_joint_delta_over_pi_pairs', 'max_joint_delta_over_0_25rad_pairs'):
                        counts[f'{name}|{phase}.{key}'] += info[phase][key]
        if RAW in arrays and CAL in arrays:
            delta = np.abs(arrays[RAW] - arrays[CAL])
            row['raw_calibrated_max_component_difference'] = float(np.nanmax(delta))
            accumulated['raw_vs_calibrated|max_component_difference'].append(np.nanmax(delta, axis=1))
        if task == 'soft' and RAW in arrays:
            mismatch = np.any(arrays['action'].astype(np.float32) != arrays[RAW].astype(np.float32), axis=1)
            counts['main_action_differs_from_raw_target_float32_rows'] += int(mismatch.sum())
        if CMD in arrays and CANDIDATE in arrays:
            mask = np.isfinite(arrays[CMD]).all(axis=1) & np.isfinite(arrays[CANDIDATE]).all(axis=1)
            if sent is not None:
                mask &= sent
            delta = np.abs(arrays[CMD][mask] - arrays[CANDIDATE][mask])
            accumulated['command_vs_candidate_sent|max_joint_difference_rad'].append(np.max(delta, axis=1))
        rows.append(row)
    return rows, {
        'task': task, 'dataset': str(root), 'frame_stride': stride,
        'scope': 'all original frames and all episode partitions; audit only, no statistics applied to training',
        'threshold_note': '30 degrees or 100 mm per pair are review flags, not automatic invalidation; stride-3 covers all three offsets',
        'counts': dict(counts), 'environment_episode_counts': dict(env_counts),
        'field_episode_counts': dict(field_episodes), 'flag_counts': dict(flag_counts),
        'distributions': {key: distribution(np.concatenate(values)) for key, values in accumulated.items()},
        'cross_episode_initial_quaternions': {key: cross_episode_signs(values) for key, values in initials.items()},
        'examples': dict(examples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output_root / f'action_audit_real_97_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    print('REPORT_DIR', output.resolve(), flush=True)
    for task, (folder, stride) in TASKS.items():
        rows, summary = audit(args.datasets_root / folder, task, stride)
        (output / f'{task}_episodes.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
        (output / f'{task}_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        compact = {k: summary[k] for k in ('task', 'counts', 'flag_counts', 'field_episode_counts', 'cross_episode_initial_quaternions')}
        compact['distributions'] = summary['distributions']
        compact['examples'] = {key: value[:2] for key, value in summary['examples'].items() if value}
        print(json.dumps(compact, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
