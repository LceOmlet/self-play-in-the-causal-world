"""Build the DoLens Figure 1 vector PDF from paper/figures/overview.html.

Reproducible: headless Chrome print-to-PDF at the exact canvas size, local fonts
only, no network. Emits paper/figures/overview.pdf plus a 220 dpi inspection PNG.

    python scripts/build_overview_figure.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
SRC = FIG_DIR / "overview.html"
PDF = FIG_DIR / "overview.pdf"
PNG = FIG_DIR / "overview-preview.png"

# Canvas size in CSS px; the PDF is emitted at 1 px -> 1/96 in.
# Included at \linewidth = 5.5in, so 1 px = 0.33 pt on the printed page.
CANVAS_W, CANVAS_H = 1200, 776

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
    raise SystemExit("No Chrome/Chromium found for headless PDF export.")


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"command failed: {cmd[0]}")


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing figure source: {SRC}")

    # Geometry gate: never export a figure that measures as clipped or colliding.
    check = ROOT / "scripts" / "check_overview_layout.py"
    if check.exists():
        proc = subprocess.run(
            [sys.executable, str(check)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit("layout check failed; figure not exported")

    chrome = find_chrome()
    with tempfile.TemporaryDirectory() as tmp:
        run(
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=10000",
                f"--user-data-dir={tmp}",
                f"--window-size={CANVAS_W},{CANVAS_H}",
                f"--print-to-pdf={PDF}",
                SRC.resolve().as_uri(),
            ]
        )

    if not PDF.exists():
        raise SystemExit("Chrome produced no PDF.")

    info = subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout
    fonts = subprocess.run(["pdffonts", str(PDF)], capture_output=True, text=True).stdout

    pages = [ln for ln in info.splitlines() if ln.startswith("Pages:")]
    size = [ln for ln in info.splitlines() if ln.startswith("Page size:")]
    print(json.dumps({"pdf": str(PDF), "pages": pages, "size": size}, indent=2))
    print(fonts)

    if pages and pages[0].split()[-1] != "1":
        raise SystemExit("figure PDF is not exactly one page")
    if "Type 3" in fonts:
        raise SystemExit("figure PDF contains a Type 3 font")
    for line in fonts.splitlines()[2:]:
        if line.strip() and line.split()[-3] != "yes":
            raise SystemExit(f"font not embedded: {line}")

    run(
        [
            "pdftoppm",
            "-png",
            "-f",
            "1",
            "-l",
            "1",
            "-singlefile",
            "-r",
            "220",
            str(PDF),
            str(PNG.with_suffix("")),
        ]
    )
    print(f"preview: {PNG}")


if __name__ == "__main__":
    main()
