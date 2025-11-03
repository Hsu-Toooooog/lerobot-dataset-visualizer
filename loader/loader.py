from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
import io
import os
import numpy as np
import pyarrow.parquet as pq
import glob
import threading

@dataclass(frozen=True)
class FrameState:
    positions: np.ndarray  # shape (J,)
    gripper: float


class LazyImage:
    """A tiny lazy image wrapper that decodes bytes on first access.

    Decoding prefers libjpeg-turbo (turbojpeg) if available, else falls back to Pillow.
    The decoded ndarray is cached per instance to avoid repeated work.
    """

    __slots__ = ("_bytes", "_array", "_lock")

    def __init__(self, b: bytes) -> None:
        self._bytes: bytes = b
        self._array: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def to_numpy(self) -> Optional[np.ndarray]:
        if self._array is not None:
            return self._array
        with self._lock:
            if self._array is not None:
                return self._array
            arr = _decode_image_fast(self._bytes)
            self._array = arr
            return self._array

    # Back-compat: some renderers may treat object as ndarray directly
    # Provide a light-weight conversion on str() to avoid accidental large repr
    def __repr__(self) -> str:
        return f"LazyImage(bytes={len(self._bytes)})"


@dataclass(frozen=True)
class FrameRGB:
    # Values are lazy (LazyImage) to avoid decoding during load; UI should call .to_numpy()
    images: Dict[str, Union[np.ndarray, LazyImage]]


@dataclass(frozen=True)
class Frame:
    timestamp: Optional[float]
    state: FrameState
    rgb: FrameRGB


@dataclass(frozen=True)
class Episode:
    frames: List[Frame]

    @property
    def length(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class LerobotDataset:
    joint_names: List[str]
    gripper_names: List[str]
    camera_names: List[str]
    episodes: List[Episode]

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def episode_lengths(self) -> List[int]:
        return [ep.length for ep in self.episodes]

    @property
    def num_frames_total(self) -> int:
        return sum(self.episode_lengths)

    def stack_positions(self) -> Tuple[np.ndarray, List[int]]:
        joint_num = len(self.joint_names)
        arrays: List[np.ndarray] = []
        for ep in self.episodes:
            if ep.length == 0:
                raise ValueError(f"Episode {self.episodes.index(ep)} has no frames.")
            ep_arr = np.stack([f.state.positions for f in ep.frames], axis=0)
            if ep_arr.shape[1] != joint_num:
                raise ValueError("positions dimension mismatch with joint number")
            arrays.append(ep_arr)
        return np.concatenate(arrays, axis=0), self.episode_lengths

    def stacked_gripper(self) -> Tuple[np.ndarray, List[int]]:
        """将所有 episode 的夹爪值堆叠为 (N,) 并返回每个 episode 的长度。"""
        values: List[float] = []
        for ep in self.episodes:
            values.extend([float(f.state.gripper) for f in ep.frames])
        return np.asarray(values, dtype=np.float32), self.episode_lengths

def load_dataset(path: str) -> LerobotDataset:
    root = os.fspath(path)
    if os.path.isdir(root):
        files = glob.glob(os.path.join(root,"**/*.parquet"), recursive=True)
        files = sorted({os.path.abspath(f) for f in files})
    else:
        raise FileNotFoundError(f"Path is not a dir: {path}")
    if files == []:
        raise FileNotFoundError(f"No parquet files found under: {path}")

    episodes: List[Episode] = []
    joint_names: Optional[List[str]] = None
    gripper_names = ["gripper"] # TODO
    camera_names_set: set[str] = set()

    first_table = pq.read_table(files[0])
    for col in first_table.column_names:
        if col.startswith("observation.images."):
            cam = col.split(".")[-1]
            camera_names_set.add(cam)
    camera_names = sorted(camera_names_set) if camera_names_set else []

    for fp in files:
        table = pq.read_table(fp)

        # Extract state vectors
        if "observation.state" not in table.column_names:
            # Skip files without expected schema
            continue
        state_list = table.column("observation.state").to_pylist()

        # Optional timestamp
        timestamps = (
            table.column("timestamp").to_pylist() if "timestamp" in table.column_names else [None] * len(state_list)
        )

        # Extract image structs per camera if present
        cam_columns: Dict[str, List[Optional[Dict[str, object]]]] = {}
        for cam in camera_names:
            name = f"observation.images.{cam}"
            if name in table.column_names:
                cam_columns[cam] = table.column(name).to_pylist()
            else:
                cam_columns[cam] = [None] * len(state_list)

        frames: List[Frame] = []
        # Infer joints count from first vector
        if not state_list:
            episodes.append(Episode(frames=[]))
            continue
        try:
            vec0 = np.array(state_list[0], dtype=np.float32)
            if vec0.ndim != 1:
                raise ValueError
            if vec0.size < 1:
                raise ValueError
            pos_dim = max(0, vec0.size - 1)
        except Exception:
            # Fallback: treat all as positions and gripper=0
            pos_dim = len(state_list[0])

        if joint_names is None:
            joint_names = [f"joint_{i+1}" for i in range(pos_dim)]

        for i in range(len(state_list)):
            s = np.array(state_list[i], dtype=np.float32)
            if s.size == 0:
                positions = np.zeros((pos_dim,), dtype=np.float32)
                gr = 0.0
            else:
                if s.size <= pos_dim:
                    positions = s.astype(np.float32)
                    gr = 0.0
                else:
                    positions = s[:pos_dim].astype(np.float32)
                    gr = float(s[7]) # TODO

            # Build RGB dict (store bytes lazily as LazyImage)
            rgb_images: Dict[str, Union[np.ndarray, LazyImage]] = {}
            for cam in camera_names:
                entry = cam_columns.get(cam, [None])[i]
                if isinstance(entry, dict):
                    b = entry.get("bytes") if isinstance(entry.get("bytes"), (bytes, bytearray, memoryview)) else None
                    if isinstance(b, memoryview):
                        b = b.tobytes()
                    if isinstance(b, (bytes, bytearray)) and len(b) > 0:
                        # defer decoding
                        rgb_images[cam] = LazyImage(bytes(b))

            frames.append(
                Frame(
                    timestamp=float(timestamps[i]) if timestamps[i] is not None else None,
                    state=FrameState(positions=positions, gripper=gr),
                    rgb=FrameRGB(images=rgb_images),
                )
            )

        episodes.append(Episode(frames=frames))

    if joint_names is None:
        joint_names = []

    return LerobotDataset(
        joint_names=joint_names,
        gripper_names=gripper_names,
        camera_names=camera_names,
        episodes=episodes,
    )


# Decoder utilities: prefer TurboJPEG, fallback to Pillow
_TJ: Optional[object] = None
_TJ_INITED = False

def _get_turbojpeg() -> Optional[object]:
    global _TJ, _TJ_INITED
    if _TJ_INITED:
        return _TJ
    try:
        from turbojpeg import TurboJPEG  # type: ignore
        _TJ = TurboJPEG()
    except Exception:
        _TJ = None
    _TJ_INITED = True
    return _TJ

def _decode_image_fast(b: Optional[bytes]) -> Optional[np.ndarray]:
    if not b:
        return None
    # Try TurboJPEG
    tj = _get_turbojpeg()
    if tj is not None:
        try:
            import numpy as _np
            from turbojpeg import TJPF_RGB  # type: ignore

            buf = _np.frombuffer(b, dtype=_np.uint8)
            arr = tj.decode(buf, pixel_format=TJPF_RGB)
            # Ensure contiguous uint8 HWC
            return _np.ascontiguousarray(arr)
        except Exception:
            pass
    # Fallback to Pillow
    try:
        from PIL import Image  # type: ignore
        with Image.open(io.BytesIO(b)) as im:
            return np.array(im.convert("RGB"))
    except Exception:
        return None