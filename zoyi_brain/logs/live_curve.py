#!/usr/bin/env python3
"""sft_25d 终端实时 loss 曲线（SSH 可用，零依赖）。

用法: python3 logs/live_curve.py     # Ctrl+C 退出
"""

import math
import re
import shutil
import sys
import time

LOG = sys.argv[1] if len(sys.argv) > 1 else "/home/zoyi/Projects/manipulation/zoyi_brain/logs/sft_open_gate_25d_state.log"
TOTAL_STEPS = 20000
STEPS_PER_EPOCH = 1345
LINE_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}\.\d{3}).*?step=(\d+) loss=([\d.eE+-]+) lr=([\d.eE+-]+) grad_norm=([\d.eE+-]+)"
)
# tqdm 进度条每步都有 loss/lr（比结构化日志密），无 grad_norm
BAR_RE = re.compile(r"loss=([\d.eE+-]+), lr=([\d.eE+-]+), step=(\d+)\]")
BLOCKS = " ▁▂▃▄▅▆▇█"


def parse():
    pts = {}  # step -> (ts, loss, lr, gnorm)
    ts_now = 0
    try:
        with open(LOG, errors="replace") as f:
            for line in f:
                m = LINE_RE.match(line)
                if m:
                    t = time.strptime(m.group(1)[:8], "%H:%M:%S")
                    ts_now = t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec
                    pts[int(m.group(2))] = (ts_now, float(m.group(3)), float(m.group(4)), float(m.group(5)))
                    continue
                b = BAR_RE.search(line)
                if b:
                    s = int(b.group(3))
                    if s not in pts:
                        pts[s] = (ts_now, float(b.group(1)), float(b.group(2)), float("nan"))
    except FileNotFoundError:
        pass
    steps = sorted(pts)
    return [(pts[s][0], s, pts[s][1], pts[s][2], pts[s][3]) for s in steps]


def chart(vals, width, height, color):
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return ["(无数据)"]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        hi = lo + 1
    n = len(vals)
    step = max(1, n // width)
    cols = [vals[i : i + step] for i in range(0, n, step)][-width:]
    rows = []
    for r in range(height, 0, -1):
        line = ""
        for col in cols:
            v = sum(col) / len(col)
            frac = (v - lo) / (hi - lo) * height
            filled = frac - (r - 1)
            idx = max(0, min(8, int(filled * 8)))
            line += BLOCKS[idx]
        rows.append(f"{color}{line}\033[0m")
    return rows


def main():
    while True:
        data = parse()
        w = shutil.get_terminal_size().columns - 12
        out = []
        if not data:
            out.append("等待日志…（训练启动后自动出曲线）")
        else:
            ts, steps, loss, lr, gn = zip(*data)
            w10 = 10
            smooth = [sum(loss[max(0, i - w10 + 1): i + 1]) / min(i + 1, w10) for i in range(len(loss))]
            dt = (ts[-1] - ts[max(0, len(ts) - 20)]) / max(1, min(20, len(ts) - 1))
            eta = (TOTAL_STEPS - steps[-1]) * dt / 3600
            gnf = [g for g in gn if math.isfinite(g)]
            gn_txt = f"{gnf[-1]:.2f}" if gnf else "—"
            gn_max = f"{max(gnf):.2f}" if gnf else "—"
            out.append(
                f"\033[1mstep {steps[-1]}/{TOTAL_STEPS}\033[0m · epoch {steps[-1]/STEPS_PER_EPOCH:.1f} · "
                f"loss \033[33m{loss[-1]:.4f}\033[0m · lr {lr[-1]:.2e} · grad_norm {gn_txt} · "
                f"{dt:.1f}s/步 · ETA {eta:.1f}h"
            )
            out.append("")
            out.append("\033[1m── loss（蓝=原始 橙=平滑10） " + "─" * max(0, w - 20) + "\033[0m")
            raw = chart(list(loss)[-w:], w, 10, "\033[36m")
            sm = chart(list(smooth)[-w:], w, 10, "\033[33m")
            for a, b in zip(raw, sm):
                out.append(a + " " + b)
            out.append(f"loss 范围: {min(loss):.4f} ~ {max(loss):.4f}")
            out.append("")
            out.append("\033[1m── grad_norm（红） / lr（绿）" + "─" * max(0, w - 20) + "\033[0m")
            for a, b in zip(chart(gnf[-w:], w, 6, "\033[31m"), chart([v for v in lr if math.isfinite(v)][-w:], w, 6, "\033[32m")):
                out.append(a + " " + b)
            gnf = [g for g in gn if math.isfinite(g)]
            gn_txt = f"{gnf[-1]:.2f}" if gnf else "—"
            gn_max = f"{max(gnf):.2f}" if gnf else "—"
            out.append(f"grad_norm 最近: {gn_txt} | 峰值: {gn_max}")
        print("\033[H\033[J" + "\n".join(out), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\033[0m已退出")
