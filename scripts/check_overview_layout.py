"""Measure the rendered geometry of paper/figures/overview.html and fail on defects.

Headless Chrome renders the figure with its real local fonts, then reports the
bounding box of every text node and container. Defects are decided from measured
pixels, not from source coordinates, because font metrics move text.

Checks: canvas overflow, container containment (`data-in`), text/text collision,
container/container collision, and minimum type size.

    python scripts/check_overview_layout.py
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
SRC = FIG_DIR / "overview.html"

CANVAS_W, CANVAS_H = 1200, 776
MIN_PX = 19.5  # ~6.4 pt once the canvas is scaled to \linewidth
EDGE_TOL = 1.0
MIN_OVERLAP_AREA = 12.0
SHRINK_Y = 0.20  # ignore ascender/descender slack when testing collisions
SHRINK_X = 0.04

PROBE = """
<script>
(async () => {
  await document.fonts.ready;
  const boxes = {};
  document.querySelectorAll('[data-box]').forEach(el => {
    const r = el.getBoundingClientRect();
    boxes[el.getAttribute('data-box')] =
      {x: r.left, y: r.top, w: r.width, h: r.height};
  });
  const items = [];
  document.querySelectorAll('text, [data-audit]').forEach((el, i) => {
    const r = el.getBoundingClientRect();
    if (r.width < 0.5 && r.height < 0.5) return;
    items.push({
      i, id: el.id || '', cls: el.getAttribute('class') || '',
      tag: el.tagName.toLowerCase(),
      box: el.getAttribute('data-in') || '',
      pad: parseFloat(el.getAttribute('data-pad') || 'NaN'),
      ov: el.getAttribute('data-ov') || '',
      fs: parseFloat(getComputedStyle(el).fontSize) || 0,
      txt: (el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 58)
           || ('<' + el.tagName.toLowerCase() + ' '
               + (el.getAttribute('data-audit') || '') + '>'),
      x: r.left, y: r.top, w: r.width, h: r.height,
    });
  });
  const json = JSON.stringify({boxes, items});
  const b64 = btoa(unescape(encodeURIComponent(json)));
  document.body.insertAdjacentHTML('beforeend',
    '<pre id="__audit__">__AUDIT_B64__' + b64 + '__END__</pre>');
})();
</script>
"""

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


def find_chrome() -> Path:
    for c in CHROME_CANDIDATES:
        if c.exists():
            return c
    for name in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return Path(found)
    raise SystemExit("No Chrome/Chromium found for layout measurement.")


def measure() -> dict:
    """Render a probe copy of the figure beside the original so fonts resolve."""
    html = SRC.read_text(encoding="utf-8")
    if "</body>" not in html:
        raise SystemExit("figure source has no </body>")
    probed = html.replace("</body>", PROBE + "</body>")

    tmp_html = SRC.with_name("__layout_probe__.html")
    tmp_html.write_text(probed, encoding="utf-8")
    chrome = find_chrome()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    str(chrome),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=12000",
                    f"--user-data-dir={tmp}",
                    f"--window-size={CANVAS_W},{CANVAS_H}",
                    "--dump-dom",
                    tmp_html.resolve().as_uri(),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        dom = proc.stdout or ""
        gate = re.search(r"<title>(FONT-ERROR[^<]*)</title>", dom)
        if gate:
            raise SystemExit(f"figure font gate failed: {gate.group(1)}")
        m = re.search(r"__AUDIT_B64__([A-Za-z0-9+/=]+)__END__", dom)
        if not m:
            sys.stderr.write(proc.stderr[-2000:])
            raise SystemExit("layout probe produced no measurements")
        return json.loads(base64.b64decode(m.group(1)).decode("utf-8"))
    finally:
        tmp_html.unlink(missing_ok=True)


def label(it: dict) -> str:
    return f"[{it['cls'] or it['id'] or 'text'}] {it['txt']!r}"


def inter_area(a: dict, b: dict, sx: float, sy: float) -> float:
    ax0, ax1 = a["x"] + a["w"] * sx, a["x"] + a["w"] * (1 - sx)
    ay0, ay1 = a["y"] + a["h"] * sy, a["y"] + a["h"] * (1 - sy)
    bx0, bx1 = b["x"] + b["w"] * sx, b["x"] + b["w"] * (1 - sx)
    by0, by1 = b["y"] + b["h"] * sy, b["y"] + b["h"] * (1 - sy)
    dx = min(ax1, bx1) - max(ax0, bx0)
    dy = min(ay1, by1) - max(ay0, by0)
    return dx * dy if dx > 1.0 and dy > 1.0 else 0.0


def audit(data: dict) -> list[str]:
    boxes, items = data["boxes"], data["items"]
    bad: list[str] = []

    for it in items:
        if it["x"] < -EDGE_TOL or it["y"] < -EDGE_TOL:
            bad.append(f"OFF-CANVAS (top/left) {label(it)} at ({it['x']:.1f},{it['y']:.1f})")
        if it["x"] + it["w"] > CANVAS_W + EDGE_TOL:
            bad.append(f"CLIPPED right by {it['x'] + it['w'] - CANVAS_W:.1f}px {label(it)}")
        if it["y"] + it["h"] > CANVAS_H + EDGE_TOL:
            bad.append(f"CLIPPED bottom by {it['y'] + it['h'] - CANVAS_H:.1f}px {label(it)}")
        if it.get("tag") == "text" and 0 < it["fs"] < MIN_PX:
            bad.append(f"TYPE TOO SMALL {it['fs']:.1f}px {label(it)}")

    for it in items:
        key = it["box"]
        if not key:
            continue
        if key not in boxes:
            bad.append(f"UNKNOWN container {key!r} for {label(it)}")
            continue
        b = boxes[key]
        pad = it["pad"]
        pad = 4.0 if pad is None or pad != pad else float(pad)
        over = [
            ("left", b["x"] + pad - it["x"]),
            ("top", b["y"] + pad - it["y"]),
            ("right", it["x"] + it["w"] - (b["x"] + b["w"] - pad)),
            ("bottom", it["y"] + it["h"] - (b["y"] + b["h"] - pad)),
        ]
        for side, amount in over:
            if amount > 0.75:
                bad.append(f"OVERFLOWS {key} on {side} by {amount:.1f}px {label(it)}")

    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if a["ov"] and a["ov"] == b["ov"]:
                continue
            area = inter_area(a, b, SHRINK_X, SHRINK_Y)
            if area > MIN_OVERLAP_AREA:
                bad.append(f"TEXT COLLISION {area:.0f}px^2 {label(a)} vs {label(b)}")

    names = sorted(boxes)
    for i, na in enumerate(names):
        for nb in names[i + 1 :]:
            if na.split(":")[0] != "field" or nb.split(":")[0] != "field":
                continue
            if inter_area(boxes[na], boxes[nb], 0.0, 0.0) > 1.0:
                bad.append(f"FIELD COLLISION {na} vs {nb}")
    return bad


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing figure source: {SRC}")
    data = measure()
    bad = audit(data)
    n_text = len(data["items"])
    if bad:
        print(f"layout FAILED: {len(bad)} defect(s) over {n_text} measured nodes\n")
        for line in bad:
            print("  - " + line)
        raise SystemExit(1)
    print(
        f"layout OK: {n_text} text nodes, {len(data['boxes'])} containers, "
        f"canvas {CANVAS_W}x{CANVAS_H}, no clipping or collisions"
    )


if __name__ == "__main__":
    main()
