"""Secondary global appearance metrics for future-video predictions."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from wan_video_action.evaluation.io import read_video_frames
from wan_video_action.evaluation.manifest import EvaluationRecord


def _align_frames(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = min(len(gt), len(pred))
    if count <= 0:
        raise ValueError("GT and prediction have no overlapping frames.")
    gt = gt[:count]
    pred = pred[:count]
    if pred.shape[1:3] != gt.shape[1:3]:
        height, width = gt.shape[1:3]
        if cv2 is not None:
            pred = np.stack([
                cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                for frame in pred
            ])
        else:
            from PIL import Image

            pred = np.stack([
                np.asarray(
                    Image.fromarray(frame).resize(
                        (width, height),
                        resample=Image.Resampling.BICUBIC,
                    )
                )
                for frame in pred
            ])
    return gt.astype(np.float32) / 255.0, pred.astype(np.float32) / 255.0


def frame_psnr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    mse = np.mean((gt - pred) ** 2, axis=(1, 2, 3))
    return np.where(mse == 0, np.inf, -10.0 * np.log10(mse))


def frame_ssim(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError(
            "SSIM evaluation requires the optional 'scikit-image' package."
        ) from exc
    return np.asarray([
        structural_similarity(
            gt_frame,
            pred_frame,
            channel_axis=-1,
            data_range=1.0,
        )
        for gt_frame, pred_frame in zip(gt, pred)
    ], dtype=np.float64)


class LPIPSEvaluator:
    def __init__(self, net: str = "alex", device: str = "cuda") -> None:
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS evaluation requires the optional 'lpips' package."
            ) from exc
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = lpips.LPIPS(
            net=net,
            lpips=True,
            pretrained=True,
            verbose=False,
        ).to(self.device).eval()

    def __call__(
        self, gt: np.ndarray, pred: np.ndarray, batch_size: int = 8
    ) -> np.ndarray:
        values: list[np.ndarray] = []
        torch = self._torch
        with torch.no_grad():
            for start in range(0, len(gt), batch_size):
                stop = min(start + batch_size, len(gt))
                gt_tensor = torch.from_numpy(gt[start:stop]).permute(0, 3, 1, 2)
                pred_tensor = torch.from_numpy(pred[start:stop]).permute(0, 3, 1, 2)
                gt_tensor = gt_tensor.to(self.device) * 2.0 - 1.0
                pred_tensor = pred_tensor.to(self.device) * 2.0 - 1.0
                batch = self.model(gt_tensor, pred_tensor)
                values.append(batch.reshape(-1).cpu().numpy())
        return np.concatenate(values)


def evaluate_global_record(
    record: EvaluationRecord,
    *,
    compute_psnr: bool = True,
    lpips_evaluator: LPIPSEvaluator | None = None,
    lpips_batch_size: int = 8,
) -> dict[str, Any]:
    gt = read_video_frames(
        record.gt_video_path,
        start_frame=record.gt_start_frame,
        num_frames=record.num_frames,
    )
    pred = read_video_frames(
        record.pred_video_path,
        start_frame=record.pred_start_frame,
        num_frames=record.num_frames,
    )
    gt, pred = _align_frames(gt, pred)
    output: dict[str, Any] = {
        "sample_id": record.sample_id,
        "environment_id": record.environment_id,
        "method": record.method,
        "split": record.split,
        "domain": record.domain,
        "support_size": record.support_size,
        "seed": record.seed,
        "evaluated_frames": len(gt),
    }
    if compute_psnr:
        output["psnr"] = float(np.mean(frame_psnr(gt, pred)))
    if lpips_evaluator is not None:
        output["lpips"] = float(np.mean(
            lpips_evaluator(gt, pred, batch_size=lpips_batch_size)
        ))
    return output
