"""Publish a complete LeRobot dataset with labels and linked source videos."""

from pathlib import Path
import json
import shutil
import tempfile

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from submodules.contracts import sha256


def _stats(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return dict(
        min=x.min(0).tolist(), max=x.max(0).tolist(), mean=x.mean(0).tolist(), std=x.std(0).tolist(), count=[len(x)]
    )


def export_training_dataset(destination, scores, frame_files, horizon=50):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Training export already exists: {destination}; choose a new directory")
    selected = scores.loc[scores.horizon == horizon]
    if selected.empty or selected.duplicated(["dataset_id", "episode_index", "frame_index"]).any():
        raise ValueError("Missing horizon or duplicate score keys")
    if set(selected.dataset_id) != {key.rsplit(":", 1)[0] for key in frame_files}:
        raise ValueError("Incomplete training export: an input dataset has no labels")
    roots = {}
    for dataset in selected.dataset_id.unique():
        if not isinstance(dataset, str) or Path(dataset).name != dataset or dataset in (".", ".."):
            raise ValueError("dataset_id must be a plain directory name")
        entries = [entry for key, entry in frame_files.items() if key.rsplit(":", 1)[0] == dataset]
        candidates = {Path(e["root"]).resolve() for e in entries}
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous source root for {dataset}")
        roots[str(dataset)] = candidates.pop()
    plans = []
    # Validate complete coverage before creating any published output.
    for dataset, root in roots.items():
        info = json.loads((root / "meta/info.json").read_text())
        files = sorted((root / "data").rglob("episode_*.parquet"))
        frame_count = 0
        seen = set()
        episode_lengths = {}
        for file in files:
            table = pq.read_table(file)
            ep = table.column("episode_index").to_numpy()
            frames = table.column("frame_index").to_numpy()
            if not len(ep) or len(set(ep)) != 1 or not np.array_equal(frames, np.arange(len(ep))):
                raise ValueError(f"Invalid source frame indices: {file}")
            episode = int(ep[0])
            if episode in seen:
                raise ValueError(f"Duplicate source episode: {dataset}:{episode}")
            seen.add(episode)
            episode_lengths[episode] = len(ep)
            expected_file = root / info["data_path"].format(
                episode_chunk=episode // info["chunks_size"], episode_index=episode
            )
            if expected_file.resolve() != file.resolve():
                raise ValueError(f"Source data_path metadata does not match storage: {file}")
            for key in ("observation.joint_positions", "action.joint_positions"):
                shape = info["features"][key]["shape"]
                values = np.asarray(table.column(key).to_pylist())
                if len(shape) != 1 or values.shape != (len(ep), shape[0]) or not np.isfinite(values).all():
                    raise ValueError(f"Source dimension metadata does not match storage: {file}:{key}")
            entry = frame_files.get(f"{dataset}:{episode}")
            if entry is None or Path(entry["file"]).resolve() != file.resolve():
                raise ValueError(f"Incomplete training export: unscored source episode {dataset}:{episode}")
            rows = selected.loc[(selected.dataset_id == dataset) & (selected.episode_index == episode)].sort_values(
                "frame_index"
            )
            if not np.array_equal(rows.frame_index.to_numpy(), frames):
                raise ValueError(f"Incomplete/misaligned labels: {dataset}:{episode}")
            if rows.is_advantage.isna().any() or not all(isinstance(v, (bool, np.bool_)) for v in rows.is_advantage):
                raise ValueError("Training labels must be non-null booleans")
            if not np.isfinite(rows.advantage_continuous.to_numpy(np.float64)).all():
                raise ValueError("Non-finite advantage values")
            for key, feature in info["features"].items():
                if feature["dtype"] != "video":
                    continue
                video = root / info["video_path"].format(
                    episode_chunk=episode // info["chunks_size"], episode_index=episode, video_key=key
                )
                cap = cv2.VideoCapture(str(video))
                try:
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if (
                        not cap.isOpened()
                        or count != len(frames)
                        or abs(cap.get(cv2.CAP_PROP_FPS) - info["fps"]) > 0.05
                    ):
                        raise ValueError(f"Missing/invalid video or frame count: {video}")
                    for index in sorted({0, len(frames) // 2, len(frames) - 1}):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                        if not cap.read()[0]:
                            raise ValueError(f"Cannot decode video frame {index}: {video}")
                finally:
                    cap.release()
            plans.append((dataset, file, rows))
            frame_count += len(ep)
        if len(seen) != info["total_episodes"] or frame_count != info["total_frames"]:
            raise ValueError(f"Source metadata counts do not match storage: {root}")
        if set(selected.loc[selected.dataset_id == dataset, "episode_index"]) != seen:
            raise ValueError(f"Scores contain unknown episodes: {dataset}")
        episodes_path = root / "meta/episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(episodes_path)
        episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
        if len(episodes) != len(seen) or any(
            e["episode_index"] not in seen or e.get("length") != episode_lengths[e["episode_index"]] for e in episodes
        ):
            raise ValueError(f"Episode metadata does not match storage: {root}")
        video_count = len(seen) * sum(f["dtype"] == "video" for f in info["features"].values())
        if info.get("total_videos", video_count) != video_count:
            raise ValueError(f"Video metadata count does not match storage: {root}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".labeled-", dir=destination.parent))
    identity = {}
    try:
        for dataset, root in roots.items():
            target = temporary / dataset
            shutil.copytree(root / "data", target / "data")
            shutil.copytree(root / "meta", target / "meta")
            (target / "videos").symlink_to((root / "videos").resolve(), target_is_directory=True)
            identity[dataset] = dict(
                source=str(root),
                frame_files={str(p.relative_to(root)): sha256(p) for p in sorted((root / "data").rglob("*.parquet"))},
            )
        labels_by_dataset = {key: [] for key in roots}
        for dataset, source, rows in plans:
            table = pq.read_table(source)
            columns = dict(
                advantage=rows.advantage_continuous.to_numpy(np.float32),
                advantage_positive=rows.is_advantage.to_numpy(bool),
            )
            for key, values in columns.items():
                if key in table.column_names:
                    table = table.set_column(table.column_names.index(key), key, pa.array(values))
                else:
                    table = table.append_column(key, pa.array(values))
            pq.write_table(table, temporary / dataset / source.relative_to(roots[dataset]))
            labels_by_dataset[dataset].append((int(rows.episode_index.iloc[0]), columns))
        for dataset, parts in labels_by_dataset.items():
            target = temporary / dataset
            info_path = target / "meta/info.json"
            info = json.loads(info_path.read_text())
            for key, dtype in (("advantage", "float32"), ("advantage_positive", "bool")):
                info["features"][key] = dict(dtype=dtype, shape=[1], names=None)
            info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
            stats_path = target / "meta/episodes_stats.jsonl"
            if stats_path.exists():
                stats = [json.loads(line) for line in stats_path.read_text().splitlines() if line]
                lookup = dict(parts)
                for episode in stats:
                    for key, values in lookup[episode["episode_index"]].items():
                        episode["stats"][key] = _stats(values)
                stats_path.write_text("".join(json.dumps(e) + "\n" for e in stats))
            aggregate = target / "meta/stats.json"
            if aggregate.exists():
                stats = json.loads(aggregate.read_text())
                for key in ("advantage", "advantage_positive"):
                    stats[key] = _stats(np.concatenate([columns[key] for _, columns in parts]))
                aggregate.write_text(json.dumps(stats, indent=2))
            manifest = dict(
                status="complete",
                horizon=horizon,
                source=identity[dataset],
                frames=info["total_frames"],
                episodes=info["total_episodes"],
                videos_linked=True,
                files={
                    str(p.relative_to(target)): sha256(p)
                    for folder in ("data", "meta")
                    for p in sorted((target / folder).rglob("*"))
                    if p.is_file()
                },
            )
            (target / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
        (temporary / "training_manifest.json").write_text(
            json.dumps(
                dict(status="complete", horizon=horizon, datasets=[str(destination / key) for key in roots]), indent=2
            )
        )
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return [str(destination / key) for key in roots]
