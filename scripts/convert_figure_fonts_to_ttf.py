"""Convert the figure's TeX Gyre OTF (CFF) faces to TrueType.

Chrome's print-to-PDF rasterizes CFF/OTF outlines into Type 3 fonts, which
conferences reject. Re-expressing the same outlines as quadratic TrueType glyphs
makes Chrome embed a real TrueType face instead. Glyph shapes, advance widths and
names are preserved; only the outline representation changes.

    python scripts/convert_figure_fonts_to_ttf.py
"""

from __future__ import annotations

from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "paper" / "figures" / "fonts"

# Max error in font units when approximating cubics with quadratics.
MAX_ERR = 1.0


def convert(src: Path, dst: Path) -> None:
    otf = TTFont(src)
    glyph_order = otf.getGlyphOrder()
    glyph_set = otf.getGlyphSet()
    upm = otf["head"].unitsPerEm

    glyphs = {}
    metrics = {}
    for name in glyph_order:
        g = glyph_set[name]
        pen = TTGlyphPen(None)
        g.draw(Cu2QuPen(pen, MAX_ERR * upm / 1000.0, reverse_direction=True))
        glyphs[name] = pen.glyph()
        metrics[name] = (g.width, 0)

    fb = FontBuilder(upm, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(otf.getBestCmap())
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)

    hhea, os2, head, post = otf["hhea"], otf["OS/2"], otf["head"], otf["post"]
    fb.setupHorizontalHeader(ascent=hhea.ascent, descent=hhea.descent, lineGap=hhea.lineGap)
    fb.setupNameTable(
        {
            n: str(otf["name"].getDebugName(i) or "")
            for n, i in (
                ("familyName", 1),
                ("styleName", 2),
                ("uniqueFontIdentifier", 3),
                ("fullName", 4),
                ("version", 5),
                ("psName", 6),
                ("copyright", 0),
                ("licenseDescription", 13),
                ("licenseInfoURL", 14),
            )
        }
    )
    fb.setupOS2(
        sTypoAscender=os2.sTypoAscender,
        sTypoDescender=os2.sTypoDescender,
        sTypoLineGap=os2.sTypoLineGap,
        usWinAscent=os2.usWinAscent,
        usWinDescent=os2.usWinDescent,
        sxHeight=getattr(os2, "sxHeight", 0),
        sCapHeight=getattr(os2, "sCapHeight", 0),
        usWeightClass=os2.usWeightClass,
        fsSelection=os2.fsSelection,
        fsType=os2.fsType,
        achVendID=os2.achVendID,
    )
    fb.setupPost(
        italicAngle=post.italicAngle,
        underlinePosition=post.underlinePosition,
        underlineThickness=post.underlineThickness,
    )
    fb.font["head"].macStyle = head.macStyle
    if "kern" in otf:
        fb.font["kern"] = otf["kern"]
    fb.save(dst)
    print(f"{src.name} -> {dst.name}  ({len(glyphs)} glyphs)")


def main() -> None:
    for src in sorted(FONT_DIR.glob("*.otf")):
        convert(src, src.with_suffix(".ttf"))


if __name__ == "__main__":
    main()
