"""Shared opt-in helpers; model imports are delayed until a compute-node call."""
from __future__ import annotations
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def require_compute():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this operation in a Slurm compute allocation, not on a login node")


class SilverDetector:
    def __init__(self):
        require_compute()
        # The CPU evaluation venv reuses these installed packages read-only.
        sys.path.append(str(ROOT / ".venv/lib/python3.10/site-packages"))
        import torch
        from transformers import AutoConfig, AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.config = json.loads((ROOT / "configs/evaluation/real97_soft_open_vocab_probe.json").read_text())
        self.filter = json.loads((ROOT / "configs/evaluation/real97_soft_open_vocab_color_filter.json").read_text())
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
        torch.set_num_interop_threads(1)
        kwargs = dict(cache_dir=str(ROOT / self.config["cache_dir"]), local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(self.config["model_id"], **kwargs)
        mc = AutoConfig.from_pretrained(self.config["model_id"], **kwargs)
        mc.disable_custom_kernels = True
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config["model_id"], config=mc, use_safetensors=True, trust_remote_code=False, **kwargs).to("cpu").eval()
        self.threshold = ("threshold" if "threshold" in inspect.signature(
            self.processor.post_process_grounded_object_detection).parameters else "box_threshold")

    def __call__(self, image):
        import cv2
        from PIL import Image
        from wan_video_action.real97_eval.open_vocab_filter import filter_proposals
        inputs = self.processor(images=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                                text=self.config["prompt"], return_tensors="pt", size=self.config["detector_resize"])
        with self.torch.inference_mode():
            raw = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(raw, inputs.input_ids, **{
            self.threshold: self.config["box_threshold"], "text_threshold": self.config["text_threshold"],
            "target_sizes": [image.shape[:2]]})[0]
        boxes = []
        for box, score in zip(result["boxes"], result["scores"]):
            x0, y0, x1, y1 = map(float, box.tolist())
            boxes.append(dict(box=[x0,y0,x1,y1], center=[(x0+x1)/2,(y0+y1)/2], score=float(score)))
        filtered = filter_proposals(image, boxes, self.filter)
        from track_real97_soft_dino_cpu import select_detection
        association = json.loads((ROOT / "configs/evaluation/real97_soft_dino_fullvideo.json").read_text())
        selected, reason = select_detection(filtered, None, 0, association)
        if selected is None:
            return dict(center=None, measurement_valid=False, reason=reason, proposals=filtered)
        x0,y0,x1,y1 = selected["box"]
        h,w = image.shape[:2]
        center = [(max(0,x0)+min(w,x1))/2,(max(0,y0)+min(h,y1))/2]
        return dict(center=center, measurement_valid=True, box=selected["box"], confidence=selected["score"],
                    reason="visible_box", proposals=filtered)


def native_frame(video, index):
    import cv2
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot decode {video} frame {index}")
    return frame
