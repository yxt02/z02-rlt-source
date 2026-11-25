#!/usr/bin/env python3
"""sft_25d 训练实时曲线面板（零依赖）。

用法: python3 loss_dashboard.py [port]   # 默认 8765
浏览器打开 http://<机器IP>:8765 （SSH 隧道: ssh -L 8765:localhost:8765 <机器>）
"""

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = sys.argv[2] if len(sys.argv) > 2 else "/home/zoyi/Projects/manipulation/zoyi_brain/logs/sft_open_gate_25d_state.log"
STEPS_PER_EPOCH = 1955  # 125133 帧 / batch 64
TOTAL_STEPS = 20000

LINE_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}\.\d{3}).*?step=(\d+) loss=([\d.eE+-]+) lr=([\d.eE+-]+) grad_norm=([\d.eE+-]+)"
)


def parse():
    steps, loss, lr, gnorm, ts = [], [], [], [], []
    try:
        with open(LOG, errors="replace") as f:
            for line in f:
                m = LINE_RE.match(line)
                if m:
                    t = time.strptime(m.group(1)[:8], "%H:%M:%S")
                    ts.append(t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec)
                    steps.append(int(m.group(2)))
                    loss.append(float(m.group(3)))
                    lr.append(float(m.group(4)))
                    gnorm.append(float(m.group(5)))
    except FileNotFoundError:
        pass
    return steps, loss, lr, gnorm, ts


def stats():
    steps, loss, lr, gnorm, ts = parse()
    if not steps:
        return {"steps": [], "loss": [], "lr": [], "gnorm": [], "step_time": 0, "eta": "等待日志…"}
    # 平滑 loss（10 步均值）
    w = 10
    smooth = [sum(loss[max(0, i - w + 1) : i + 1]) / min(i + 1, w) for i in range(len(loss))]
    # 步速：取最近 20 个日志时间戳差分
    step_time = 0.0
    if len(ts) >= 2:
        dt = (ts[-1] - ts[max(0, len(ts) - 20)]) / max(1, min(20, len(ts) - 1))
        if dt > 0:
            step_time = dt
    remain = (TOTAL_STEPS - steps[-1]) * step_time
    eta = f"{remain/3600:.1f} 小时" if remain > 0 else "完成"
    return {
        "steps": steps, "loss": loss, "smooth": smooth, "lr": lr, "gnorm": gnorm,
        "cur_step": steps[-1], "cur_loss": loss[-1],
        "epoch": round(steps[-1] / STEPS_PER_EPOCH, 1),
        "step_time": round(step_time, 1), "eta": eta,
    }


HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>sft_25d 训练曲线</title>
<style>
body{background:#111;color:#ddd;font-family:monospace;margin:20px}
.hdr{font-size:18px;margin-bottom:8px}
canvas{background:#1b1b1b;border:1px solid #333;margin:6px 0;width:100%;height:260px}
.badge{display:inline-block;background:#222;border:1px solid #444;border-radius:4px;padding:2px 10px;margin-right:10px}
</style></head><body>
<div class="hdr">pi0.5 sft_25d · batch=64 · 实时监控</div>
<div id="hdr"></div>
<canvas id="c1"></canvas><canvas id="c2"></canvas>
<script>
function draw(cv, series, logscale){
  const c=document.getElementById(cv), x=c.getContext('2d');
  c.width=c.clientWidth*2; c.height=c.clientHeight*2; x.scale(2,2);
  const W=c.clientWidth,H=c.clientHeight; x.clearRect(0,0,W,H);
  let all=series.flatMap(s=>s.data).filter(v=>isFinite(v)&&(!logscale||v>0));
  if(!all.length){x.fillStyle='#666';x.fillText('暂无数据',10,20);return;}
  let lo=Math.min(...all),hi=Math.max(...all);
  const map=v=>logscale?Math.log10(Math.max(v,1e-12)):v;
  let llo=map(lo),lhi=map(hi); if(lhi-llo<1e-9){lhi=llo+1;}
  series.forEach(s=>{
    x.beginPath(); x.strokeStyle=s.color; x.lineWidth=1.5;
    s.data.forEach((v,i)=>{
      if(!isFinite(v)||(logscale&&v<=0))return;
      const px=i/(s.data.length-1||1)*(W-60)+50;
      const py=H-20-(map(v)-llo)/(lhi-llo)*(H-40);
      i===0?x.moveTo(px,py):x.lineTo(px,py);
    }); x.stroke();
    x.fillStyle=s.color; x.fillText(s.name,55,14+series.indexOf(s)*14);
  });
  x.fillStyle='#888'; x.fillText(hi.toExponential(2),2,24); x.fillText(lo.toExponential(2),2,H-24);
}
function refresh(){
  fetch('/data').then(r=>r.json()).then(d=>{
    document.getElementById('hdr').innerHTML=
      `<span class="badge">step ${d.cur_step||0} / ${''' + str(TOTAL_STEPS) + '''}</span>`+
      `<span class="badge">epoch ${d.epoch||0}</span>`+
      `<span class="badge">loss ${d.cur_loss?d.cur_loss.toFixed(4):'-'}</span>`+
      `<span class="badge">${d.step_time||'-'} s/step</span>`+
      `<span class="badge">ETA ${d.eta}</span>`;
    draw('c1',[{name:'loss (raw)',data:d.loss,color:'#5af'},{name:'loss (smooth 10)',data:d.smooth,color:'#fa5'}],false);
    draw('c2',[{name:'lr (log)',data:d.lr,color:'#5f5'},{name:'grad_norm (log)',data:d.gnorm,color:'#f55'}],true);
  });
}
refresh(); setInterval(refresh, 5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            body = json.dumps(stats()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"dashboard: http://0.0.0.0:{port}  (log: {LOG})")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
