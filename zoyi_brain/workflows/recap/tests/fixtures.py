"""Small real LeRobot v2 dataset with three decodable cameras; no external weights."""

from pathlib import Path
import json

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def make_dataset(root, dim=25, episodes=8, frames=4):
    root = Path(root)
    (root / "meta").mkdir(parents=True)
    features = {
        key: dict(dtype="float32", shape=[dim], names=[f"joint_{i}" for i in range(dim)])
        for key in ("observation.joint_positions", "action.joint_positions")
    }
    for key in ("timestamp", "reward"):
        features[key] = dict(dtype="float32", shape=[1], names=None)
    for key in ("episode_index", "frame_index", "index", "task_index", "intervention"):
        features[key] = dict(dtype="int64", shape=[1], names=None)
    for cam in ("cam2", "cam3", "cam4"):
        features[f"observation.images.{cam}"] = dict(
            dtype="video",
            shape=[16, 24, 3],
            names=["height", "width", "channels"],
            info={
                "video.fps": 10,
                "video.height": 16,
                "video.width": 24,
                "video.channels": 3,
                "video.codec": "mpeg4",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        )
    info = dict(
        codebase_version="v2.0",
        robot_type="z02",
        fps=10,
        chunks_size=1000,
        total_episodes=episodes,
        total_frames=episodes * frames,
        total_tasks=1,
        total_chunks=1,
        total_videos=episodes * 3,
        splits={"train": f"0:{episodes}"},
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        features=features,
    )
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(json.dumps(dict(task_index=0, task="Pick up the cup")) + "\n")
    records = []
    for episode in range(episodes):
        state = np.full((frames, dim), 0.3, dtype=np.float32)
        state[:, :16] = np.arange(frames)[:, None] * 0.1 + episode * 0.01
        state[:, [7, 15]] = np.arange(frames)[:, None] % 2
        if dim > 24:
            state[:, 24] = 0
            if episode == 0:
                state[-1, 24] = 0.2
        reward = np.full(frames, -1.0, np.float32)
        reward[-1] = 0 if episode % 2 == 0 else -2000
        action = state.copy()
        action[:, :7] += 0.05
        action[:, 8:15] += 0.05
        table = pa.table(
            {
                "observation.joint_positions": state.tolist(),
                "action.joint_positions": action.tolist(),
                "episode_index": [episode] * frames,
                "frame_index": np.arange(frames),
                "index": np.arange(frames) + episode * frames,
                "task_index": [0] * frames,
                "timestamp": np.arange(frames) / 10,
                "reward": reward,
                "intervention": np.arange(frames) % 2,
            }
        )
        file = root / info["data_path"].format(episode_chunk=0, episode_index=episode)
        file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, file)
        for camera in ("cam2", "cam3", "cam4"):
            video = root / info["video_path"].format(
                episode_chunk=0, episode_index=episode, video_key=f"observation.images.{camera}"
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (24, 16))
            if not writer.isOpened():
                raise RuntimeError("OpenCV test video encoder unavailable")
            for frame in range(frames):
                writer.write(np.full((16, 24, 3), frame * 40 + episode, np.uint8))
            writer.release()
        records.append(dict(episode_index=episode, tasks=["Pick up the cup"], length=frames))
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in records))
    stats = {}
    for key, feature in features.items():
        shape = feature["shape"] if feature["dtype"] != "video" else [3, 1, 1]
        stats[key] = dict(
            min=np.zeros(shape).tolist(),
            max=np.ones(shape).tolist(),
            mean=np.zeros(shape).tolist(),
            std=np.ones(shape).tolist(),
            count=[frames * episodes],
        )
    (root / "meta/stats.json").write_text(json.dumps(stats))
    return root
