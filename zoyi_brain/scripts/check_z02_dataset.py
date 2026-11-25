#!/usr/bin/env python3
"""Z02 新数据集体检：专查录制管线反复出现的元数据 bug + 基本完整性。

已知 bug 清单（每次新数据都要查）：
  1. meta/info.json 里 state/action 维度写成 23，实际是 25
  2. 4 个幽灵 observation.forces.* 特征（数据里没有对应列）
  3. parquet 里 joint_positions 是变长 list/double 而非定长 float32[25]
  4. 视频/相机键缺失（cam2=head, cam3=hand_right, cam4=hand_left）

用法（zoyi_brain 目录下）：
  .venv/bin/python scripts/check_z02_dataset.py <dataset_dir> [<dataset_dir> ...]
"""

import json
import pathlib
import sys

import pyarrow.parquet as pq

EXPECTED_JOINT_DIM = 25
CAM_KEYS = ["observation.images.cam2", "observation.images.cam3", "observation.images.cam4"]
JOINT_KEYS = ["observation.joint_positions", "action.joint_positions"]


def check_dataset(root: pathlib.Path) -> int:
    print(f"\n===== {root} =====")
    problems = 0
    if not root.exists():
        print("❌ 目录不存在（数据可能还在传输）")
        return 1

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        print("❌ 缺少 meta/info.json")
        return 1
    info = json.loads(info_path.read_text())
    features = info.get("features", {})

    # 1. 关节维度
    for key in JOINT_KEYS:
        feat = features.get(key)
        if feat is None:
            print(f"❌ info.json 缺特征 {key}")
            problems += 1
            continue
        shape = feat.get("shape")
        dim = shape[0] if isinstance(shape, list) and shape else shape
        if dim != EXPECTED_JOINT_DIM:
            print(f"❌ {key} 维度={dim}，应为 {EXPECTED_JOINT_DIM}（已知 bug：写成 23）")
            problems += 1
        else:
            print(f"✅ {key} 维度={dim}")

    # 2. 幽灵 forces 特征（info.json 声明但数据里不存在才算幽灵）
    ghosts = [k for k in features if "forces" in k]
    if ghosts:
        print(f"ℹ️ forces 特征 {len(ghosts)} 个: {ghosts}（是否幽灵以 parquet 实际列为准，见第 4 步）")
    else:
        print("✅ 无 forces 特征")

    # 3. 相机键
    for key in CAM_KEYS:
        if key in features:
            print(f"✅ {key} 存在")
        else:
            print(f"❌ 缺相机特征 {key}")
            problems += 1

    # 4. parquet 类型与实际数据
    data_dir = root / "data" / "chunk-000"
    files = sorted(data_dir.glob("episode_*.parquet")) if data_dir.exists() else []
    if not files:
        print(f"❌ 没有找到数据 parquet（data/chunk-000/episode_*.parquet）")
        return problems + 1
    table = pq.read_table(files[0])
    schema_names = set(table.schema.names)
    for key in JOINT_KEYS:
        if key not in schema_names:
            print(f"❌ parquet 缺列 {key}")
            problems += 1
            continue
        col = table.column(key)
        first = col[0].as_py()
        lens = {len(v) for v in col.to_pylist()[:50]}
        ok_len = lens == {EXPECTED_JOINT_DIM}
        typ = table.schema.field(key).type
        print(f"{'✅' if ok_len else '❌'} parquet {key}: type={typ} 样本长度={sorted(lens)}")
        if not ok_len:
            problems += 1
        if not isinstance(first, list):
            print(f"❌ {key} 不是 list 结构")
            problems += 1
    missing = [k for k in features if k not in schema_names and "images" not in k and "timestamp" not in k and "frame_index" not in k and "episode_index" not in k and "index" not in k and "task" not in k]
    if missing:
        print(f"⚠️ info.json 声明但 parquet 没有的列（幽灵特征）: {missing}")
        problems += len(missing)

    # 5. 规模与任务
    n_episodes = len(list((root / "videos").glob("chunk-000/observation.images.cam2/*"))) if (root / "videos").exists() else None
    print(f"📁 data parquet 数: {len(files)}  (≈episode 数)  videos.cam2 episode 数: {n_episodes}")
    total_frames = info.get("total_frames") or info.get("total_frames", "?")
    print(f"📁 total_frames={info.get('total_frames', '?')}  total_episodes={info.get('total_episodes', '?')}  fps={info.get('fps', '?')}")
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks = pq.read_table(tasks_path).to_pylist()
        print(f"📝 tasks: {[t.get('task') for t in tasks][:5]}")

    print(f"{'✅ 通过' if problems == 0 else f'❌ 发现 {problems} 个问题'}")
    return problems


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    total = 0
    for arg in sys.argv[1:]:
        total += check_dataset(pathlib.Path(arg).expanduser())
    print(f"\n===== 总计问题数: {total} =====")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
