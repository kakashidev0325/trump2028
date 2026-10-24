"""Build Trump 2028 site images from the hat logo and the podium banner.

Memes are photo types (caption, split, labels, comic, badge, film strip)
made only from those two pictures.
"""
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np
import cv2
import os

ROOT = r"E:\pack\solana\trump 2028"
FONT = os.path.join(ROOT, "assets", "LilitaOne-Regular.ttf")
SRC_LOGO = os.path.join(ROOT, "logo.png")
SRC_BANNER = os.path.join(ROOT, "banner.png")

INK = (27, 31, 75, 255)
YELLOW = (255, 210, 31, 255)
PINK = (255, 45, 149, 255)
PAPER = (251, 247, 239, 255)
WHITE = (255, 255, 255, 255)
RED = (186, 22, 36, 255)
BLACK = (8, 8, 12, 255)


def font(size):
    return ImageFont.truetype(FONT, size)


def text_size(draw, text, fnt, sw=0):
    box = draw.textbbox((0, 0), text, font=fnt, stroke_width=sw)
    return box[2] - box[0], box[3] - box[1], box


def stamp(draw, xy, text, fnt, fill, stroke=BLACK, sw=8):
    draw.text(xy, text, font=fnt, fill=fill, stroke_width=sw, stroke_fill=stroke)


def stamp_center(draw, y, text, fnt, width, fill, stroke=BLACK, sw=8):
    tw, th, box = text_size(draw, text, fnt, sw)
    x = (width - tw) // 2 - box[0]
    stamp(draw, (x, y - box[1]), text, fnt, fill, stroke, sw)
    return th


def stamp_center_in(draw, x0, x1, y, text, fnt, fill, stroke=BLACK, sw=0):
    tw, th, box = text_size(draw, text, fnt, sw)
    x = x0 + (x1 - x0 - tw) // 2 - box[0]
    stamp(draw, (x, y - box[1]), text, fnt, fill, stroke, sw)


def word(base, xy, text, fnt, fill=YELLOW, sw=10, shift=(7, 7)):
    draw = ImageDraw.Draw(base)
    x, y = xy
    draw.text((x + shift[0], y + shift[1]), text, font=fnt, fill=(55, 16, 18, 255))
    draw.text((x, y), text, font=fnt, fill=fill, stroke_width=sw, stroke_fill=INK)
    return base


def word_center(base, y, text, fnt, fill=YELLOW, sw=10, shift=(7, 7)):
    draw = ImageDraw.Draw(base)
    tw, th, box = text_size(draw, text, fnt, sw)
    x = (base.width - tw) // 2 - box[0]
    word(base, (x, y), text, fnt, fill, sw, shift)
    return th


def save_webp(im, path, quality=86):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, "WEBP", quality=quality, method=6)


def cover(im, w, h, focus=(0.5, 0.45)):
    im = im.convert("RGB")
    ratio = max(w / im.width, h / im.height)
    nw, nh = max(1, int(im.width * ratio)), max(1, int(im.height * ratio))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    x = int((nw - w) * focus[0])
    y = int((nh - h) * focus[1])
    x = max(0, min(nw - w, x))
    y = max(0, min(nh - h, y))
    return im.crop((x, y, x + w, y + h))


def contain_pad(im, w, h, bg=(12, 48, 140)):
    im = im.convert("RGB")
    ratio = min(w / im.width, h / im.height)
    nw, nh = int(im.width * ratio), int(im.height * ratio)
    fitted = im.resize((nw, nh), Image.Resampling.LANCZOS)
    blur = im.resize((w, h), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(22))
    overlay = Image.new("RGB", (w, h), bg)
    blur = Image.blend(blur, overlay, 0.28)
    blur.paste(fitted, ((w - nw) // 2, (h - nh) // 2))
    return blur


def trim(im, pad=8):
    bbox = im.split()[-1].getbbox()
    if not bbox:
        return im
    x0, y0, x1, y1 = bbox
    return im.crop((
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(im.width, x1 + pad),
        min(im.height, y1 + pad),
    ))


def fit(im, w, h, scale=0.92, valign="center"):
    im = trim(im, 4)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    max_w, max_h = int(w * scale), int(h * scale)
    ratio = min(max_w / im.width, max_h / im.height)
    nw, nh = max(1, int(im.width * ratio)), max(1, int(im.height * ratio))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    x = (w - nw) // 2
    if valign == "end":
        y = h - nh - int(h * (1 - scale) / 2)
    elif valign == "start":
        y = int(h * (1 - scale) / 2)
    else:
        y = (h - nh) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def knock_white(im):
    """Drop the white studio field. Lettering stays, because it is enclosed by red."""
    rgb = np.array(im.convert("RGB"))
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    seeds = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 2), (w // 2, h - 3), (2, h // 2), (w - 3, h // 2),
        (40, h - 8), (w - 40, h - 8), (80, 8), (w - 80, 8),
    ]
    for seed in seeds:
        if mask[seed[1] + 1, seed[0] + 1] == 0:
            cv2.floodFill(bgr, mask, seed, 0, (42, 42, 42), (42, 42, 42), flags)
    alpha = np.where(mask[1:-1, 1:-1] > 0, 0, 255).astype(np.uint8)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    pale = (mx > 205) & ((mx.astype(np.int16) - mn.astype(np.int16)) < 36)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(8):
        edge = cv2.dilate((alpha < 16).astype(np.uint8), kernel) & (alpha > 16)
        alpha = np.where(edge.astype(bool) & pale, 0, alpha).astype(np.uint8)
    r16, g16, b16 = r.astype(np.int16), g.astype(np.int16), b.astype(np.int16)
    red = (r16 > 110) & (r16 > g16 + 35) & (r16 > b16 + 35)
    row_i = np.arange(h, dtype=np.int16)[:, None]
    lowest = np.where(red, row_i, -1).max(axis=0)
    below = row_i > (lowest[None, :] + 1)
    desat = (mx.astype(np.int16) - mn.astype(np.int16)) < 55
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha = np.where(below & desat & (lowest[None, :] >= 0), 0, alpha).astype(np.uint8)
    rgba = im.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    return rgba


def key_blue(im):
    """Drop the blue hall behind the podium. Shirt, suit, and cap stay."""
    arr = np.array(im.convert("RGBA"))
    rgb = arr[:, :, :3].astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    blue = (b > 120) & (b > r + 35) & (b > g + 8) & (r < 170)
    alpha = np.where(blue, 0, 255).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha = cv2.erode(alpha, np.ones((2, 2), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr)


def shade_bars(im, top=180, bot=210, strength=200):
    im = im.convert("RGBA")
    h, w = im.size
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    px = overlay.load()
    for i in range(top):
        a = int(strength * (1 - i / top) ** 1.35)
        for x in range(w):
            px[x, i] = (0, 0, 0, a)
    for i in range(bot):
        a = int(strength * (1 - i / bot) ** 1.15)
        y = h - 1 - i
        for x in range(w):
            px[x, y] = (0, 0, 0, a)
    return Image.alpha_composite(im, overlay)


def circle_photo(im, size):
    tile = cover(im, size, size, (0.5, 0.42)).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    tile.putalpha(mask)
    return tile


def frame_photo(src, w, h, focus=(0.5, 0.45)):
    src = src.convert("RGB")
    ratio = min(w / src.width, h / src.height)
    nw, nh = max(1, int(src.width * ratio)), max(1, int(src.height * ratio))
    fitted = src.resize((nw, nh), Image.Resampling.LANCZOS)
    blur = cover(src, w, h, focus).filter(ImageFilter.GaussianBlur(14))
    dim = Image.new("RGB", (w, h), (12, 14, 28))
    blur = Image.blend(blur, dim, 0.35)
    blur.paste(fitted, ((w - nw) // 2, (h - nh) // 2))
    return blur


def meme_caption(logo):
    im = shade_bars(cover(logo, 1080, 1080, (0.5, 0.42)), 210, 250, 215)
    d = ImageDraw.Draw(im)
    stamp_center(d, 42, "TRUMP", font(120), 1080, WHITE, sw=14)
    stamp_center(d, 900, "2028", font(110), 1080, WHITE, sw=14)
    stamp_center(d, 1018, "$TRUMP2028", font(36), 1080, YELLOW, sw=6)
    return im.convert("RGB")


def meme_split(logo, podium):
    w, h = 1280, 720
    gap = 10
    left_w = (w - gap) // 2
    im = Image.new("RGB", (w, h), (27, 31, 75))
    im.paste(cover(logo, left_w, h, (0.5, 0.42)), (0, 0))
    im.paste(cover(podium, w - left_w - gap, h, (0.5, 0.42)), (left_w + gap, 0))
    im = im.convert("RGBA")
    bar = Image.new("RGBA", im.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(bar)
    bd.rounded_rectangle((28, 620, 390, 688), radius=18, fill=(8, 8, 12, 210))
    bd.rounded_rectangle((w - 430, 620, w - 28, 688), radius=18, fill=(8, 8, 12, 210))
    bd.rounded_rectangle((w // 2 - 190, 24, w // 2 + 190, 96), radius=18, fill=(255, 210, 31, 245))
    im = Image.alpha_composite(im, bar)
    d = ImageDraw.Draw(im)
    stamp_center_in(d, 28, 390, 632, "THE CAP", font(36), WHITE, sw=0)
    stamp_center_in(d, w - 430, w - 28, 632, "THE PODIUM", font(34), WHITE, sw=0)
    stamp_center_in(d, w // 2 - 190, w // 2 + 190, 38, "$TRUMP2028", font(36), INK, sw=0)
    return im.convert("RGB")


def meme_labels(banner):
    im = cover(banner, 1152, 864, (0.5, 0.35)).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    calls = [
        ((28, 24, 340, 108), (500, 175), "THE CAP"),
        ((28, 730, 360, 822), (548, 390), "THE MIC"),
        ((800, 24, 1124, 108), (640, 210), "2028"),
    ]
    for (x0, y0, x1, y1), (tx, ty), _label in calls:
        ax, ay = (x0 + x1) // 2, (y0 + y1) // 2
        if ty > y1:
            ay = y1
        elif ty < y0:
            ay = y0
        d.line(((ax, ay), (tx, ty)), fill=(255, 255, 255, 230), width=6)
        d.ellipse((tx - 8, ty - 8, tx + 8, ty + 8), fill=YELLOW)
        d.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=(8, 8, 12, 220))
    im = Image.alpha_composite(im, overlay)
    d = ImageDraw.Draw(im)
    stamp_center_in(d, 28, 340, 40, "THE CAP", font(36), YELLOW, sw=0)
    stamp_center_in(d, 28, 360, 748, "THE MIC", font(36), YELLOW, sw=0)
    stamp_center_in(d, 800, 1124, 40, "2028", font(40), YELLOW, sw=0)
    return im.convert("RGB")


def meme_comic(cap, podium, hall):
    w, h = 864, 1152
    im = Image.new("RGB", (w, h), (27, 31, 75))
    panels = [
        (cap, "THE CAP", (0.5, 0.4)),
        (podium, "THE PODIUM", (0.5, 0.4)),
        (hall, "THE HALL", (0.18, 0.5)),
    ]
    ph = 340
    gap = 8
    for i, (src, label, focus) in enumerate(panels):
        y = 16 + i * (ph + gap)
        tile = frame_photo(src, w - 32, ph - 64, focus)
        im.paste(tile, (16, y))
        d = ImageDraw.Draw(im)
        d.rectangle((16, y + ph - 64, w - 16, y + ph), fill=(8, 8, 12))
        stamp_center_in(d, 16, w - 16, y + ph - 52, label, font(36), YELLOW, sw=0)
    d = ImageDraw.Draw(im)
    stamp_center(d, 1068, "$TRUMP2028", font(44), w, YELLOW, sw=0)
    return im.convert("RGB")


def meme_badge(logo):
    w = h = 1000
    im = Image.new("RGBA", (w, h), RED)
    word_center(im, 36, "TRUMP", font(120), sw=12, shift=(8, 8))
    word_center(im, 168, "2028", font(78), sw=9, shift=(6, 6))
    circ = circle_photo(logo, 460)
    ring = Image.new("RGBA", im.size, (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    cx, cy = w // 2, 530
    rd.ellipse((cx - 248, cy - 248, cx + 248, cy + 248), fill=INK)
    im = Image.alpha_composite(im, ring)
    im.paste(circ, (cx - 230, cy - 230), circ)
    word_center(im, 860, "$TRUMP2028", font(58), sw=8, shift=(5, 5))
    return im.convert("RGB")


def meme_film(cap, hands, mic, hall):
    w, h = 1400, 788
    im = Image.new("RGBA", (w, h), PAPER)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w, 118), fill=RED)
    word(im, (36, 18), "THE CAP", font(58), sw=8, shift=(5, 5))
    word(im, (860, 26), "$TRUMP2028", font(44), sw=7, shift=(5, 5))
    frames = [
        (cap, "CAP", (0.5, 0.4)),
        (hands, "HANDS", (0.5, 0.42)),
        (mic, "MIC", (0.5, 0.55)),
        (hall, "HALL", (0.15, 0.5)),
    ]
    fw, fh = 300, 500
    gap = 28
    total = 4 * fw + 3 * gap
    x = (w - total) // 2
    y = 160
    for src, label, focus in frames:
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((x - 6, y - 6, x + fw + 6, y + fh + 6), radius=18, fill=INK)
        tile = cover(src, fw, fh - 70, focus).convert("RGBA")
        mask = Image.new("L", tile.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, fw, fh - 70), radius=12, fill=255)
        tile.putalpha(mask)
        im.paste(tile, (x, y), tile)
        d = ImageDraw.Draw(im)
        stamp_center_in(d, x, x + fw, y + fh - 58, label, font(36), YELLOW, sw=0)
        x += fw + gap
    return im.convert("RGB")


def thumb_for(im, tw, th):
    return cover(im, tw, th, (0.5, 0.42))


def draw_solana(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    shapes = [
        ([(64.6, 3.8), (73.8, 0), (391.2, 0), (395.8, 11.1), (333.1, 73.8), (323.9, 77.6), (6.5, 77.6), (1.9, 66.5)], (153, 69, 255, 255)),
        ([(333.1, 120.1), (323.9, 116.3), (6.5, 116.3), (1.9, 127.4), (64.6, 190.1), (73.8, 193.9), (391.2, 193.9), (395.8, 182.8)], (135, 82, 243, 255)),
        ([(64.6, 237.9), (73.8, 234.1), (391.2, 234.1), (395.8, 245.2), (333.1, 307.9), (323.9, 311.7), (6.5, 311.7), (1.9, 300.6)], (20, 241, 149, 255)),
    ]
    pad = size * 0.14
    s = min((size - 2 * pad) / 398, (size - 2 * pad) / 312)
    ox = (size - 398 * s) / 2
    oy = (size - 312 * s) / 2
    for pts, color in shapes:
        draw.polygon([(ox + x * s, oy + y * s) for x, y in pts], fill=color)
    return img


def oval_face(im, size=640, focus=(0.5, 0.45)):
    tile = cover(im, size, size, focus).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((18, 18, size - 19, size - 19), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(4))
    tile.putalpha(mask)
    return tile


def oval_tall(im, w=520, h=760, focus=(0.5, 0.45)):
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    side = int(min(w, h) * 0.92)
    face = oval_face(im, side, focus)
    canvas.paste(face, ((w - side) // 2, (h - side) // 2), face)
    return canvas


def on_red(im, w=900, h=520):
    bg = Image.new("RGBA", (w, h), RED)
    fitted = fit(im, w, h, 0.92, "center")
    bg.alpha_composite(fitted)
    return bg.convert("RGB")


def brand_tile(logo_rgb, size):
    """The hat photo, squared. The header crops it to a circle."""
    return logo_rgb.resize((size, size), Image.Resampling.LANCZOS)


def main():
    logo = Image.open(SRC_LOGO).convert("RGB")
    banner = Image.open(SRC_BANNER).convert("RGB")
    hat = knock_white(logo)
    figure = trim(key_blue(banner), 2)

    # Crops of the one podium photo, used as the other stills.
    hands = banner.crop((620, 140, 980, 430))
    mic = banner.crop((680, 180, 900, 500))
    hall = banner.crop((0, 0, 520, 500))
    cap_on = banner.crop((600, 0, 980, 280))

    brand = os.path.join(ROOT, "imagery", "brand")
    mark = brand_tile(logo, 512)
    save_webp(mark, os.path.join(brand, "logo.webp"), 90)
    mark.save(os.path.join(brand, "icon-512.png"), "PNG")
    brand_tile(logo, 192).save(os.path.join(brand, "icon-192.png"), "PNG")
    brand_tile(logo, 180).save(os.path.join(brand, "apple-touch-icon.png"), "PNG")
    brand_tile(logo, 32).save(os.path.join(brand, "favicon-32.png"), "PNG")
    brand_tile(logo, 256).save(
        os.path.join(ROOT, "favicon.ico"), sizes=[(16, 16), (32, 32), (48, 48), (64, 64)]
    )
    contain_pad(banner, 1200, 630).save(os.path.join(brand, "og.jpg"), "JPEG", quality=88, optimize=True)
    save_webp(draw_solana(256), os.path.join(brand, "sol.webp"), 90)

    scenes = os.path.join(ROOT, "imagery", "scenes")
    save_webp(banner, os.path.join(scenes, "hero.webp"), 86)
    save_webp(mark.resize((500, 500), Image.Resampling.LANCZOS), os.path.join(scenes, "mark.webp"), 88)
    save_webp(banner, os.path.join(scenes, "season2.webp"), 86)

    trump = os.path.join(ROOT, "imagery", "trump")
    pieces = {
        "head": oval_tall(logo, focus=(0.5, 0.4)),
        "love": oval_tall(cap_on, focus=(0.5, 0.45)),
        "x": oval_tall(hall, focus=(0.72, 0.45)),
        "body": fit(figure, 520, 760, 0.98, "end"),
        "sit": fit(figure, 480, 760, 0.96, "end"),
        "dex": fit(hat, 480, 760, 0.9, "center"),
        "x-bite": fit(hat, 640, 640, 0.9),
        "dex-bite": oval_face(logo, 640, (0.5, 0.42)),
    }
    for name, im in pieces.items():
        save_webp(im, os.path.join(trump, name + ".webp"))

    memes = {
        1: meme_caption(logo),
        2: meme_split(logo, banner),
        3: meme_labels(banner),
        4: meme_comic(logo, banner, banner),
        5: meme_badge(logo),
        6: meme_film(logo, hands, mic, hall),
    }
    thumbs = {1: (640, 640), 2: (1138, 640), 3: (853, 640), 4: (480, 640), 5: (640, 640), 6: (1138, 640)}
    meme_dir = os.path.join(ROOT, "imagery", "memes")
    thumb_dir = os.path.join(meme_dir, "thumbs")
    for n, img in memes.items():
        save_webp(img, os.path.join(meme_dir, f"{n}.webp"), 86)
        tw, th = thumbs[n]
        save_webp(thumb_for(img, tw, th), os.path.join(thumb_dir, f"{n}.webp"), 80)
    # The rail used to ship twenty posters. Later numbers repeat these six types.
    cycle = {7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 6, 19: 1, 20: 2}
    for n, src_n in cycle.items():
        img = memes[src_n]
        save_webp(img, os.path.join(meme_dir, f"{n}.webp"), 86)
        tw, th = thumbs[src_n]
        save_webp(thumb_for(img, tw, th), os.path.join(thumb_dir, f"{n}.webp"), 80)

    prev = os.path.join(ROOT, "assets")
    on_red(hat, 900, 520).save(os.path.join(prev, "preview-cutouts.jpg"), "JPEG", quality=88)
    on_red(figure, 900, 640).save(os.path.join(prev, "preview-figure.jpg"), "JPEG", quality=88)
    memes[1].save(os.path.join(prev, "preview-meme1.jpg"), "JPEG", quality=86)
    memes[3].save(os.path.join(prev, "preview-meme3.jpg"), "JPEG", quality=86)
    memes[5].save(os.path.join(prev, "preview-meme5.jpg"), "JPEG", quality=86)
    memes[6].save(os.path.join(prev, "preview-meme6.jpg"), "JPEG", quality=86)
    print("hat", trim(hat).size, "figure", figure.size)
    print("memes", {k: v.size for k, v in memes.items()})


if __name__ == "__main__":
    main()
