import os

import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision
from PIL import Image

from diffsynth.core.data.operators import (
    DataProcessingOperator,
    FrameSamplerByRateMixin,
    LoadImage,
    RouteByType,
    SequencialProcess,
    ToList,
)
from wan_video_action.utils import align_num_frames, resolve_path

"""
Class: DataProcessingOperator
-----------------------------

Overloads the right-shift operator (`>>`) utilizing the `__rshift__` magic method.

This implementation facilitates intuitive pipeline composition, allowing multiple 
data processing operators to be chained together sequentially 
(e.g., `operator_A >> operator_B`).
"""
class RouteByKeyExtension(DataProcessingOperator):
    """
    Applies a given operator to a specific key in a dictionary.

    Args:
        key: The dictionary key containing the file path to route.
        operator_map: List of (extensions, operator) tuples for routing by file extension.
    """
    def __init__(self, key: str, operator_map=None):
        self.key = key
        self.operator_map = operator_map or []
        
    def __call__(self, data):
        path = data.get(self.key, "") if isinstance(data, dict) else data
        ext = path.split('.')[-1].lower()
        
        for exts, operator in self.operator_map:
            if ext in exts:
                return operator(data) # 传递完整上下文
                
        raise ValueError(f"Unsupported extension: {ext} for data {data}")
    
    
class ToAbsolutePathByKeyExtension(DataProcessingOperator):
    def __init__(self, base_path="", key=""):
        self.base_path = base_path
        self.key = key
        
    def __call__(self, data):
        path = data.get(self.key, "") if isinstance(data, dict) else data
        return resolve_path(self.base_path, path)


class ResolvePromptEmbPath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data: str):
        return resolve_path(self.base_path, data)


class LoadVideoChunk(DataProcessingOperator, FrameSamplerByRateMixin):
    def __init__(
        self,
        base_path="",
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
        frame_rate=24,
        fix_frame_rate=False,
        frame_stride=1,
        pad_short=False,
    ):
        FrameSamplerByRateMixin.__init__(self, num_frames, time_division_factor, time_division_remainder, frame_rate, fix_frame_rate)
        self.base_path = base_path
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.frame_stride = max(1, int(frame_stride))
        self.pad_short = bool(pad_short)

    def __call__(self, data, start_frame=None, end_frame=None):
        if isinstance(data, dict):
            path = data.get("data")
            start_frame = start_frame if start_frame is not None else data.get("start_frame")
            end_frame = end_frame if end_frame is not None else data.get("end_frame")
        else:
            raise TypeError(f"Expected 'data' to be a dict, but received {type(data).__name__}.")
            
        path = resolve_path(self.base_path, path)
            
        reader = self.get_reader(path)
        raw_frame_rate = reader.get_meta_data()['fps']
        total_raw_frames = reader.count_frames()
        
        start = max(0, start_frame if start_frame is not None else 0)
        end = min(total_raw_frames, (end_frame + 1) if end_frame is not None else total_raw_frames)
        clip_frames = max(0, end - start)
        if clip_frames <= 0:
            raise ValueError(f"No frames available in {path} for start={start_frame}, end={end_frame}.")

        # x / clip_frames = self.frame_rate / raw_frame_rate
        available_frames = (
            int(clip_frames * self.frame_rate / raw_frame_rate)
            if self.fix_frame_rate
            else (clip_frames - 1) // self.frame_stride + 1
        )
        num_frames = self.num_frames
        if available_frames < num_frames and not self.pad_short:
            num_frames = align_num_frames(
                available_frames,
                time_division_factor=self.time_division_factor,
                time_division_remainder=self.time_division_remainder,
            )
        
        frames = []
        for frame_id in range(num_frames):
            if self.fix_frame_rate:
                frame_id = self.map_single_frame_id(frame_id, raw_frame_rate, clip_frames)
            else:
                frame_id *= self.frame_stride
            frame_id = min(frame_id, clip_frames - 1)
            frame = reader.get_data(start + frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames
    

class LoadGIFChunk(DataProcessingOperator):
    def __init__(
        self,
        base_path="",
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
        pad_short=False,
    ):
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        self.pad_short = bool(pad_short)

    def get_num_frames(self, clip_frames):
        num_frames = self.num_frames
        if clip_frames < num_frames and not self.pad_short:
            num_frames = align_num_frames(
                clip_frames,
                time_division_factor=self.time_division_factor,
                time_division_remainder=self.time_division_remainder,
            )
        return num_frames
        
    def __call__(self, data, start_frame=None, end_frame=None):
        if isinstance(data, dict):
            path = data.get("data")
            start_frame = start_frame if start_frame is not None else data.get("start_frame")
            end_frame = end_frame if end_frame is not None else data.get("end_frame")
        else:
            raise TypeError(f"Expected 'data' to be a dict, but received {type(data).__name__}.")
            
        path = resolve_path(self.base_path, path)
            
        images = iio.imread(path, mode="RGB")
        total_raw_frames = len(images)
        
        start = max(0, start_frame if start_frame is not None else 0)
        end = min(total_raw_frames, (end_frame + 1) if end_frame is not None else total_raw_frames)
        clip_frames = max(0, end - start)
        if clip_frames <= 0:
            raise ValueError(f"No frames available in {path} for start={start_frame}, end={end_frame}.")

        num_frames = self.get_num_frames(clip_frames)
        frames = []
        for frame_id in range(num_frames):
            img = images[start + min(frame_id, clip_frames - 1)]
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
        return frames


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1, resize_mode: str = "fit"):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.resize_mode = resize_mode # "fit" / "crop"

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size

        if self.resize_mode == "crop":
            scale = max(target_width / width, target_height / height)
            image = torchvision.transforms.functional.resize(
                image,
                (round(height*scale), round(width*scale)),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
            return image

        elif self.resize_mode == "fit":
            image = torchvision.transforms.functional.resize(
                image, 
                [target_height, target_width], 
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
            return image
        
    def get_height_width(self, image):
        if self.resize_mode == "crop" and self.height is not None and self.width is not None:
            return self.height, self.width

        width, height = image.size
        max_area = self.height * self.width if (self.height is not None and self.width is not None) else self.max_pixels
        if max_area is not None and width * height > max_area:
            scale = (width * height / max_area) ** 0.5
            height, width = int(height / scale), int(width / scale)
        height = height // self.height_division_factor * self.height_division_factor
        width = width // self.width_division_factor * self.width_division_factor

        return height, width

    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image
    
    
class ToVideoTensor(DataProcessingOperator):
    """Convert loaded video frames to float tensor in (V, C, T, H, W), range [-1, 1].

    This operator converts a list of PIL Images or list of lists (for multi-view)
    into a normalized video tensor.
    """

    @staticmethod
    def _frame_to_tensor(frame: Image.Image) -> torch.Tensor:
        """Convert a single PIL Image to CHW tensor in range [-1, 1]."""
        if not isinstance(frame, Image.Image):
            raise TypeError(f"Expected PIL.Image, got {type(frame).__name__}")
        
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
            
        array = np.asarray(frame, dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()  # (C, H, W)
        tensor = tensor * (2.0 / 255.0) - 1.0
        return tensor

    def _frames_to_video_tensor(self, frames) -> torch.Tensor:
        """Convert a list of frames to (C, T, H, W) tensor."""
        if not isinstance(frames, (list, tuple)) or len(frames) == 0:
            raise ValueError("Expected non-empty frame list.")
        
        frame_tensors = [self._frame_to_tensor(frame) for frame in frames]
        video = torch.stack(frame_tensors, dim=1)  # (C, T, H, W)
        return video

    def __call__(self, data):
        """Convert data to video tensor.

        Args:
            data: One of:
                - torch.Tensor: Already a tensor, validate shape
                - PIL.Image: Single frame, treat as 1-frame video
                - list of PIL.Image: Single-view video
                - list of list of PIL.Image: Multi-view video

        Returns:
            torch.Tensor of shape (V, C, T, H, W) in range [-1, 1]
        """
        if isinstance(data, torch.Tensor):
            if data.ndim != 5:
                raise ValueError(f"Expected video tensor with shape (V,C,T,H,W), got {tuple(data.shape)}")
            
            return data.to(dtype=torch.float32)

        if isinstance(data, Image.Image):
            data = [data]

        if not isinstance(data, (list, tuple)) or len(data) == 0:
            raise TypeError("Expected loaded video frames as list/tuple.")

        # Check if multi-view (list of lists)
        if isinstance(data[0], (list, tuple)):
            views = [self._frames_to_video_tensor(view) for view in data]
            return torch.stack(views, dim=0)  # (V, C, T, H, W)

        # Single view
        video = self._frames_to_video_tensor(data).unsqueeze(0)  # (1, C, T, H, W)
        return video
    
# TODO: change "OBS_ACTION_NAMES" to "JOINT_AND_EEF_NAMES" 
JOINT_AND_EEF_NAMES = [
    "left_arm_joint_1_rad",
    "left_arm_joint_2_rad",
    "left_arm_joint_3_rad",
    "left_arm_joint_4_rad",
    "left_arm_joint_5_rad",
    "left_arm_joint_6_rad",
    "left_gripper_open",
    "left_eef_pos_x_m",
    "left_eef_pos_y_m",
    "left_eef_pos_z_m",
    "left_eef_rot_euler_x_rad",
    "left_eef_rot_euler_y_rad",
    "left_eef_rot_euler_z_rad",
    "right_arm_joint_1_rad",
    "right_arm_joint_2_rad",
    "right_arm_joint_3_rad",
    "right_arm_joint_4_rad",
    "right_arm_joint_5_rad",
    "right_arm_joint_6_rad",
    "right_gripper_open",
    "right_eef_pos_x_m",
    "right_eef_pos_y_m",
    "right_eef_pos_z_m",
    "right_eef_rot_euler_x_rad",
    "right_eef_rot_euler_y_rad",
    "right_eef_rot_euler_z_rad",
]

JOINT_NAMES = [
    "left_arm_joint_1_rad",
    "left_arm_joint_2_rad",
    "left_arm_joint_3_rad",
    "left_arm_joint_4_rad",
    "left_arm_joint_5_rad",
    "left_arm_joint_6_rad",
    "left_gripper_open",
    "right_arm_joint_1_rad",
    "right_arm_joint_2_rad",
    "right_arm_joint_3_rad",
    "right_arm_joint_4_rad",
    "right_arm_joint_5_rad",
    "right_arm_joint_6_rad",
    "right_gripper_open",
]

# TODO: change "POSE_NAMES" to "EEF_NAMES"
EEF_NAMES = [
    "left_eef_pos_x_m",
    "left_eef_pos_y_m",
    "left_eef_pos_z_m",
    "left_eef_rot_euler_x_rad",
    "left_eef_rot_euler_y_rad",
    "left_eef_rot_euler_z_rad",
    "left_gripper_open",
    "right_eef_pos_x_m",
    "right_eef_pos_y_m",
    "right_eef_pos_z_m",
    "right_eef_rot_euler_x_rad",
    "right_eef_rot_euler_y_rad",
    "right_eef_rot_euler_z_rad",
    "right_gripper_open",
]


class LoadCobotAction(DataProcessingOperator):
    def __init__(
        self,
        base_path="",
        action_type="eef_abs",
        stat=None,
        use_percentile_stats=True,
        num_frames=81,
        align_num_frames=True,
        time_division_factor=4,
        time_division_remainder=1,
        pad_short=False,
        output_dim=None,
        frame_stride=1,
    ):
        self.num_frames = num_frames
        self.align_num_frames = bool(align_num_frames)
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.pad_short = bool(pad_short)
        self.output_dim = None if output_dim is None else int(output_dim)
        self.frame_stride = max(1, int(frame_stride))
        """
            joint_abs (原 state_joint：关节绝对位置)
            eef_abs (原 state_pose：末端绝对位姿)
            joint_delta (原 action_joint：关节相对动作/增量)
            eef_delta (原 action_pose：末端相对动作/增量)
        """
        requested_action_type = action_type
        # Compatibility aliases for older script conventions.
        action_type_alias = {
            "joint_abs": "state_joint",
            "eef_abs": "state_pose",
            "joint_delta": "action_joint",
            "eef_delta": "action_pose",
        }
        action_type = action_type_alias.get(action_type, action_type)

        if action_type not in ("state_joint", "state_pose", "action_joint", "action_pose"):
            raise ValueError(f"Unsupported action type: {action_type}")
        self.base_path = base_path
        self.requested_action_type = requested_action_type
        self.action_type = action_type
        self.stat = stat or {}
        self.use_percentile_stats = use_percentile_stats
        # `state_*` means read observation.state and `action_*` means read action.
        self.use_state = action_type.startswith("state_")
        self.use_joint = action_type.endswith("_joint")
        name_to_idx = {name: idx for idx, name in enumerate(JOINT_AND_EEF_NAMES)}
        self.indices = [name_to_idx[name] for name in (JOINT_NAMES if self.use_joint else EEF_NAMES)]
        self._stat_min = None
        self._stat_max = None

        entry = None
        if isinstance(self.stat, dict):
            if action_type in self.stat and isinstance(self.stat[action_type], dict):
                entry = self.stat[action_type]
            elif requested_action_type in self.stat and isinstance(self.stat[requested_action_type], dict):
                entry = self.stat[requested_action_type]
            elif all(k in self.stat for k in ("min", "max")):
                # Backward-compat mode: accept direct per-type dict payload.
                entry = self.stat

        if entry is not None:
            if self.use_percentile_stats:
                # Prefer percentile stats if available; fallback to min/max for datasets
                # that do not provide p01/p99 (e.g. current RoboTwin export).
                p01 = entry.get("p01")
                p99 = entry.get("p99")
                if p01 is not None and p99 is not None:
                    self._stat_min = np.asarray(p01, dtype=np.float32)
                    self._stat_max = np.asarray(p99, dtype=np.float32)
                elif "min" in entry and "max" in entry:
                    self._stat_min = np.asarray(entry.get("min", []), dtype=np.float32)
                    self._stat_max = np.asarray(entry.get("max", []), dtype=np.float32)
            else:
                self._stat_min = np.asarray(entry.get("min", []), dtype=np.float32)
                self._stat_max = np.asarray(entry.get("max", []), dtype=np.float32)

    def _resolve_parquet_info(self, data, start_frame, end_frame):
        if isinstance(data, dict):
            parquet_rel = data.get("data")
            if start_frame is None:
                start_frame = data.get("start_frame")
            if end_frame is None:
                end_frame = data.get("end_frame")
        else:
            parquet_rel = data
        
        if not parquet_rel:
            raise KeyError("Missing parquet path in metadata 'data' field.")
        
        parquet_path = resolve_path(self.base_path, parquet_rel)

        start_frame = int(start_frame)
        end_frame = int(end_frame)
        return parquet_path, start_frame, end_frame

    def _get_min_max(self):
        if self._stat_min is not None and self._stat_max is not None:
            return self._stat_min, self._stat_max
        raise KeyError(f"Missing normalization stats for action type: {self.action_type}")

    def _normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1.0
        return np.clip(ndata, clip_min, clip_max)

    def _match_output_dim(self, arr: np.ndarray) -> np.ndarray:
        if self.output_dim is None or arr.shape[1] == self.output_dim:
            return arr
        if self.output_dim == 14 and (not self.use_joint) and arr.shape[1] == 7:
            padded = np.zeros((arr.shape[0], 14), dtype=arr.dtype)
            # BWM's 14-D EEF convention is [left_eef(7), right_eef(7)].
            padded[:, 7:] = arr
            return padded
        if self.output_dim == 14 and arr.shape[1] == 8:
            padded = np.zeros((arr.shape[0], 14), dtype=arr.dtype)
            # Single-arm Panda action followed by six neutral padding channels.
            padded[:, :8] = arr
            return padded
        raise ValueError(
            f"Cannot adapt action width {arr.shape[1]} to output_dim={self.output_dim} "
            f"for action type {self.action_type}"
        )

    def _read_slice(self, parquet_path, column, start_frame, num_frames):
        start = int(start_frame)
        end = start + int(num_frames)
        table = pq.read_table(parquet_path, columns=[column])
        data = table.to_pydict()[column]
        if end > len(data):
            if not self.pad_short:
                raise ValueError(
                    f"Not enough rows in {parquet_path} for slice "
                    f"start={start_frame}, num_frames={num_frames}"
                )
            if start >= len(data):
                raise ValueError(
                    f"No rows in {parquet_path} for padded slice "
                    f"start={start_frame}, num_frames={num_frames}"
                )
            rows = list(data[start:])
            rows.extend([rows[-1]] * (end - len(data)))
            return np.asarray(rows, dtype=np.float32)
        return np.asarray(data[start:end], dtype=np.float32)

    def get_num_frames(self, total_frames):
        if self.num_frames is None:
            return int(total_frames)
        if self.pad_short:
            return int(self.num_frames)
        num_frames = int(self.num_frames)
        if int(total_frames) < num_frames:
            num_frames = int(total_frames)
            if self.align_num_frames:
                num_frames = align_num_frames(
                    num_frames,
                    time_division_factor=self.time_division_factor,
                    time_division_remainder=self.time_division_remainder,
                )
        return num_frames

    def __call__(self, data: str, start_frame=None, end_frame=None):
        parquet_path, start_frame, end_frame = self._resolve_parquet_info(
            data, start_frame, end_frame
        )
        clip_frames = end_frame - start_frame + 1
        available_frames = (clip_frames - 1) // self.frame_stride + 1
        num_frames = self.get_num_frames(available_frames)
        column = "observation.state" if self.use_state else "action"
        raw_num_frames = min(clip_frames, max(1, (num_frames - 1) * self.frame_stride + 1))
        arr = self._read_slice(parquet_path, column, start_frame, raw_num_frames)
        if self.frame_stride > 1:
            arr = arr[::self.frame_stride]
        if arr.shape[0] < num_frames:
            if not self.pad_short:
                raise ValueError(
                    f"Not enough strided action rows in {parquet_path}: "
                    f"available={arr.shape[0]}, requested={num_frames}, stride={self.frame_stride}"
                )
            arr = np.concatenate(
                [arr, np.repeat(arr[-1:], num_frames - arr.shape[0], axis=0)],
                axis=0,
            )
        arr = arr[:num_frames]
        if arr.ndim != 2:
            raise ValueError(f"Unexpected action shape {arr.shape} in {parquet_path}")
        if arr.shape[1] == len(JOINT_AND_EEF_NAMES):
            arr = arr[:, self.indices]
        elif self.output_dim is not None and arr.shape[1] <= self.output_dim:
            # Generic single-arm LeRobot datasets can expose native action widths
            # other than the legacy 7-D EEF or 14-D bimanual conventions.
            pass
        elif self.use_joint and arr.shape[1] == len(JOINT_NAMES):
            pass
        elif (not self.use_joint) and arr.shape[1] == len(EEF_NAMES):
            pass
        elif (not self.use_joint) and arr.shape[1] == 7:
            pass
        else:
            raise ValueError(
                f"Unexpected action width {arr.shape[1]} for action type {self.action_type} in {parquet_path}"
            )
        min_vals, max_vals = self._get_min_max()
        arr = self._normalize_bound(arr, min_vals, max_vals)
        arr = self._match_output_dim(arr)
        return arr[None, ...]


def create_video_operator(
    base_path="",
    max_pixels=1920*1080, height=None, width=None,
    height_division_factor=16, width_division_factor=16,
    num_frames=81, time_division_factor=4, time_division_remainder=1,
    resize_mode="fit", default_key="data", pad_short=False, frame_stride=1,
):
    image_processor = ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, resize_mode=resize_mode)
    
    image_pipeline = ToAbsolutePathByKeyExtension(base_path) >> LoadImage() >> image_processor >> ToList()
    
    gif_pipeline = LoadGIFChunk(
        base_path=base_path,
        num_frames=num_frames,
        time_division_factor=time_division_factor,
        time_division_remainder=time_division_remainder,
        frame_processor=image_processor,
        pad_short=pad_short,
    )
    video_pipeline = LoadVideoChunk(
        base_path=base_path,
        num_frames=num_frames,
        time_division_factor=time_division_factor,
        time_division_remainder=time_division_remainder,
        frame_processor=image_processor,
        frame_stride=frame_stride,
        pad_short=pad_short,
    )
    
    video_operator = RouteByKeyExtension(key=default_key, operator_map=[
        (("jpg", "jpeg", "png", "webp"), image_pipeline),
        (("gif",), gif_pipeline),
        (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), video_pipeline),
    ])
    # Support dict (with metadata), str (single path), and list (multi-view)
    return RouteByType(operator_map=[
        (dict, video_operator),
        (str, video_operator),
        (list, SequencialProcess(video_operator)),
    ]) >> ToVideoTensor()
