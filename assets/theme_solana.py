"""Solana palette for the walls, and a Trump figure in the community card."""
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
import os
import importlib.util

ROOT = r"E:\pack\solana\trump 2028"
spec = importlib.util.spec_from_file_location(
    "build_images", os.path.join(ROOT, "assets", "build_images.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DEEP = (18, 8, 42)
PURPLE = (153, 69, 255)
GREEN = (20, 241, 149)


def recolor(im):
    """Move yellow and orange toward Solana green, cyan and blue toward purple."""
    rgb = np.array(im.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    h = h.astype(np.int16)
    # Masks are taken from the original hue so a pixel is shifted once.
    warm = (h >= 6) & (h <= 50) & (s > 35)      # orange / yellow
    cool = (h >= 70) & (h <= 145) & (s > 25)    # cyan / blue
    h = np.where(warm, 78, h)                   # Solana green
    h = np.where(cool, 134, h)                  # Solana purple
    h = np.clip(h, 0, 179).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
    return Image.fromarray(out)


def halftone(draw, box, color, step=14, r=2):
    x0, y0, x1, y1 = box
    for y in range(y0, y1, step):
        for x in range(x0 + (step // 2 if (y // step) % 2 else 0), x1, step):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def community_card():
    banner = Image.open(os.path.join(ROOT, "banner.png")).convert("RGB")
    figure = mod.trim(mod.key_blue(banner), 2)
    w, h = 1672, 941
    im = Image.new("RGB", (w, h), DEEP)
    draw = ImageDraw.Draw(im)
    # Right-side Solana panel, same job as the old mascot backdrop.
    draw.polygon([(860, 0), (w, 0), (w, h), (620, h)], fill=(36, 16, 84))
    draw.polygon([(1080, 0), (w, 0), (w, h), (900, h)], fill=(88, 36, 190))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    halftone(od, (980, 40, w - 20, h - 40), (20, 241, 149, 150), 16, 2)
    halftone(od, (860, 80, 1180, h - 20), (255, 255, 255, 70), 18, 2)
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    # White sweep along the bottom, where the old character leaned.
    sweep = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sweep)
    sd.polygon([(700, 760), (w, 620), (w, h), (560, h)], fill=(244, 240, 255, 230))
    sd.polygon([(760, 820), (w, 700), (w, h), (640, h)], fill=(20, 241, 149, 180))
    im = Image.alpha_composite(im.convert("RGBA"), sweep)
    target_h = int(h * 1.05)
    ratio = target_h / figure.height
    nw, nh = int(figure.width * ratio), int(figure.height * ratio)
    fig = figure.resize((nw, nh), Image.Resampling.LANCZOS)
    x = w - nw + 40
    y = h - nh + 30
    im.paste(fig, (x, y), fig)
    out = im.convert("RGB")
    path = os.path.join(ROOT, "imagery", "wall", "cool.webp")
    out.save(path, "WEBP", quality=86, method=6)
    out.resize((900, 506), Image.Resampling.LANCZOS).save(
        os.path.join(ROOT, "assets", "preview-cool.jpg"), "JPEG", quality=86
    )


def walls():
    wall = os.path.join(ROOT, "imagery", "wall")
    for name in ("burst.webp", "paper.webp", "skyline.webp", "frame.webp"):
        path = os.path.join(wall, name)
        src = Image.open(path)
        mode = src.mode
        rgb = recolor(src)
        if "A" in mode:
            rgb = rgb.convert("RGBA")
            rgb.putalpha(src.convert("RGBA").split()[-1])
            rgb.save(path, "WEBP", quality=86, method=6)
        else:
            rgb.save(path, "WEBP", quality=86, method=6)
    recolor(Image.open(os.path.join(wall, "burst.webp"))).resize((800, 266)).save(
        os.path.join(ROOT, "assets", "preview-burst.jpg"), "JPEG", quality=84
    )


def css():
    path = os.path.join(ROOT, "css", "style.css")
    text = open(path, encoding="utf-8").read()
    swaps = [
        ("#1b1f4b", "#140c2e"),
        ("#4b4f72", "#5c4d82"),
        ("#8b8da6", "#9b90b8"),
        ("#fbf7ef", "#f4f0ff"),
        ("#062867", "#1a0b3d"),
        ("#0b1f5c", "#241047"),
        ("#ffd21f", "#14F195"),
        ("#ffe46a", "#8dffd0"),
        ("#ff7a2c", "#9945FF"),
        ("#ff9a52", "#c084fc"),
        ("#f0601a", "#6d28d9"),
        ("#2c3fd8", "#14F195"),
        ("#ff2d95", "#7c3aed"),
        ("#ffe27c", "#d9ffef"),
        ("#ffe56b", "#b6ffe4"),
        ("#fff4cf", "#efe9ff"),
        ("#ffc21a", "#14F195"),
        ("#ffb51c", "#12c98a"),
        ("#ffbe1f", "#3de8a8"),
        ("#ff9ccf", "#d8b4fe"),
        ("#25306f", "#3b1d7a"),
        ("#12153a", "#0e0820"),
        ("#12307a", "#2a1460"),
        ("#0a1f57", "#1a0b40"),
        ("#05153f", "#0c0624"),
        ("#0a1a4a", "#160a32"),
        ("#0b1a47", "#14082e"),
        ("#08224f", "#14082e"),
        ("#0b2a5c", "#1a0b3d"),
        ("#2a1608", "#f4efff"),
        ("#3b2f08", "#140c2e"),
        ("#6b4708", "#140c2e"),
        ("rgba(251,247,239", "rgba(244,240,255"),
        ("rgba(27,31,75", "rgba(20,12,46"),
        ("rgba(255,210,31", "rgba(20,241,149"),
        ("rgba(255,45,149", "rgba(124,58,237"),
        ("rgba(255,154,82", "rgba(153,69,255"),
        ("rgba(255,122,44", "rgba(153,69,255"),
        ("rgba(255,190,30", "rgba(20,241,149"),
        ("rgba(16,24,70", "rgba(20,12,46"),
        ("rgba(160,100,0", "rgba(13,90,60"),
        ("rgba(44,63,216", "rgba(153,69,255"),
    ]
    for old, new in swaps:
        text = text.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(text)


if __name__ == "__main__":
    community_card()
    walls()
    css()
    print("theme done")
