"""Run the frozen multi-background protocol on the completed common-action model."""

import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

job = 121864
step = 4600
slug = "ours_mainview_8env6action_common_curr1000"
training_root = Path(
    "outputs/pushbox_matchedphysics30bg_active20/"
    "ours_c32_roi10x_mainview_8env6action_common_curr1000_lr003_2b200/121864"
)
checkpoint = training_root / f"step-{step}.safetensors"
table = training_root / f"step-{step}.context_table.json"
if not checkpoint.is_file() or checkpoint.stat().st_size < 10_000_000_000:
    raise RuntimeError(f"Missing or incomplete checkpoint: {checkpoint}")
if not table.is_file() or len(json.loads(table.read_text())["records"]) != 20:
    raise RuntimeError(f"Expected 20 environment codes in {table}")

benchmark = Path(
    "results/pushbox_multibackground/"
    "matchedphysics30bg_active20_id5_ood5_k1_oracle_informative_support25_60_v1"
)
output = benchmark / "methods" / slug / f"train_job_{job}" / f"step_{step}" / "seed_20260903"
output.mkdir(parents=True, exist_ok=True)
if (output / "evaluation.complete").is_file():
    print(f"[already complete] {output}", flush=True)
    raise SystemExit(0)

script = (ROOT / "jobs/run_eval_push_box_matchedphysics30bg_ours4400_low.sh").read_text()


def replace_line(key: str, value: str) -> None:
    global script
    script, count = re.subn(
        rf"^{re.escape(key)}=.*$", lambda _: f'{key}="{value}"', script, flags=re.MULTILINE
    )
    if count != 1:
        raise RuntimeError(f"Expected one {key} assignment in formal inference template; found {count}")


def replace_exact(old: str, new: str) -> None:
    global script
    if old not in script:
        raise RuntimeError(f"Formal inference template changed: {old}")
    script = script.replace(old, new)


replace_line(
    "CONFIG",
    "configs/train/train_push_box_matchedphysics30bg_active20_ours_c32_roi10x_mainview_8env6action_common_curr1000_lr003_2b200_24h.yaml",
)
replace_line("CKPT_SOURCE", str(checkpoint))
replace_line("TABLE_SOURCE", str(table))
replace_line("OUT", str(output))
replace_line("LOCAL_CKPT", str(checkpoint))
replace_line("LOCAL_TABLE", str(table))
replace_exact("pb30a20-ours.lock", f"{slug}-job{job}-step{step}.lock")
replace_exact(
    "protocol[key] for key in ('domain','support_indices','query_indices','target_indices','target_sample_ids','support_selection')",
    "protocol[key] for key in ('domain','support_indices','query_indices','target_indices','target_sample_ids','support_selection') if key in protocol",
)
replace_exact("f'{bench}/metrics/ours_step4400'", f"f'{{bench}}/metrics/{slug}_job{job}_step{step}'")
replace_exact("'methods':{'ours':", f"'methods':{{'{slug}':")
replace_exact('--prediction-label "Ours query"', '--prediction-label "Ours common query"')

run_script = output / "run_inference.sh"
temporary = run_script.with_suffix(".sh.tmp")
temporary.write_text(script)
temporary.replace(run_script)
(output / "checkpoint_selection.json").write_text(
    json.dumps(
        {
            "training_job_id": job,
            "checkpoint_step": step,
            "checkpoint": str(checkpoint),
            "context_table": str(table),
            "active_environment_count": 20,
        },
        indent=2,
    )
    + "\n"
)
print(f"[selected] {checkpoint}; output={output}", flush=True)
os.execv("/bin/bash", ["bash", str(run_script)])
