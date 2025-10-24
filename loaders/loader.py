from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import io
import os
import numpy as np
import streamlit as st


@dataclass(frozen=True)
class FrameState:
    positions: np.ndarray  # shape (J,)
    gripper: float


@dataclass(frozen=True)
class FrameRGB:
    images: Dict[str, np.ndarray]


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

@st.cache_resource(show_spinner=True)
def load_dataset(path: str) -> LerobotDataset:
    """Load a LeRobot-style dataset from parquet file(s).
    Behavior:
      - If 'path' is a directory, recursively finds all *.parquet and treats each file as one episode.
      - If 'path' points to a single parquet file, loads it as a single episode.
      - Splits observation.state into positions (first N-1 dims) and gripper (last dim).
      - Decodes image bytes for each available camera if Pillow is installed; otherwise leaves images as None.
    """
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("pyarrow is required to load parquet datasets. Please install pyarrow.") from e

    root = os.fspath(path)
    if os.path.isdir(root):
        # Prefer files that look like episodes, fallback to any parquet
        files = []
        for p in (
            # common patterns
            "**/episode_*.parquet",
            "**/file-*.parquet",
            "**/*.parquet",
        ):
            import glob
            candidates = glob.glob(os.path.join(root, p), recursive=True)
            files.extend(candidates)
        # de-duplicate and sort
        files = sorted({os.path.abspath(f) for f in files})
    elif os.path.isfile(root) and root.endswith(".parquet"):
        files = [os.path.abspath(root)]
    else:
        raise FileNotFoundError(f"Path not found or not a parquet: {path}")

    if not files:
        raise FileNotFoundError(f"No parquet files found under: {path}")

    def _decode_image(b: Optional[bytes]) -> Optional[np.ndarray]:
        if b is None:
            return None
        try:
            from PIL import Image  # type: ignore
        except Exception:
            return None  # Pillow not installed; leave as None
        try:
            with Image.open(io.BytesIO(b)) as im:
                return np.array(im.convert("RGB"))
        except Exception:
            return None

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

            # Build RGB dict
            rgb_images: Dict[str, np.ndarray] = {}
            for cam in camera_names:
                entry = cam_columns.get(cam, [None])[i]
                img_arr: Optional[np.ndarray] = None
                if isinstance(entry, dict):
                    b = entry.get("bytes") if isinstance(entry.get("bytes"), (bytes, bytearray, memoryview)) else None
                    if isinstance(b, memoryview):
                        b = b.tobytes()
                    img_arr = _decode_image(b)  # may remain None if Pillow missing
                if img_arr is not None:
                    rgb_images[cam] = img_arr

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