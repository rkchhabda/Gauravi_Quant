from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path

W, H = 400, 140  # Logo size - wider for horizontal layout
img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# Colors
DARK_BG = (13, 17, 23, 0)  # Transparent
GOLD = (210, 153, 34, 255)
GOLD_DARK = (160, 110, 20, 255)
GREEN = (63, 185, 80, 255)
RED = (248, 81, 73, 255)
WHITE = (230, 237, 243, 255)
MUTED = (139, 148, 158, 255)
ACCENT_BLUE = (88, 166, 255, 255)

# Font - use Windows fonts
FONT_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_REG = r"C:\Windows\Fonts\segoeui.ttf"

from PIL import ImageFont
font_big = ImageFont.truetype(FONT_BOLD, 64)
font_small = ImageFont.truetype(FONT_REG, 20)
font_tag = ImageFont.truetype(FONT_REG, 16)

# ===== 3D "G" MONOGRAM WITH CANDLESTICK ELEMENTS =====
# Draw a 3D block letter G with financial chart inside

# G shape coordinates - create a 3D extruded G
g_center_x = 70
g_center_y = H // 2
g_radius = 45
g_thickness = 14

# 3D extrusion offset
extrude = 6

# Draw extrusion shadow (bottom-right)
for i in range(extrude, 0, -1):
    alpha = int(40 * (i / extrude))
    # Outer arc
    d.arc(
        [g_center_x - g_radius + i, g_center_y - g_radius + i,
         g_center_x + g_radius + i, g_center_y + g_radius + i],
        220, 140, fill=(0, 0, 0, alpha), width=g_thickness
    )
    # Inner horizontal bar of G
    d.line(
        [(g_center_x + i, g_center_y + i), (g_center_x + g_radius - 5 + i, g_center_y + i)],
        fill=(0, 0, 0, alpha), width=g_thickness
    )
    # Vertical stem of G
    d.line(
        [(g_center_x + g_radius - 5 + i, g_center_y - g_radius + i),
         (g_center_x + g_radius - 5 + i, g_center_y + g_radius + i)],
        fill=(0, 0, 0, alpha), width=g_thickness)

# Main G shape - gradient gold
# Outer arc
for i in range(g_thickness):
    ratio = i / g_thickness
    r = int(GOLD[0] * (1 - ratio) + GOLD_DARK[0] * ratio)
    g = int(GOLD[1] * (1 - ratio) + GOLD_DARK[1] * ratio)
    b = int(GOLD[2] * (1 - ratio) + GOLD_DARK[2] * ratio)
    d.arc(
        [g_center_x - g_radius + i//2, g_center_y - g_radius + i//2,
         g_center_x + g_radius - i//2, g_center_y + g_radius - i//2],
        220, 140, fill=(r, g, b, 255), width=1
    )

# Horizontal bar of G
for i in range(g_thickness):
    ratio = i / g_thickness
    r = int(GOLD[0] * (1 - ratio) + GOLD_DARK[0] * ratio)
    g = int(GOLD[1] * (1 - ratio) + GOLD_DARK[1] * ratio)
    b = int(GOLD[2] * (1 - ratio) + GOLD_DARK[2] * ratio)
    d.line(
        [(g_center_x, g_center_y + i//2), (g_center_x + g_radius - 5, g_center_y + i//2)],
        fill=(r, g, b, 255), width=1
    )

# Vertical stem
for i in range(g_thickness):
    ratio = i / g_thickness
    r = int(GOLD[0] * (1 - ratio) + GOLD_DARK[0] * ratio)
    g = int(GOLD[1] * (1 - ratio) + GOLD_DARK[1] * ratio)
    b = int(GOLD[2] * (1 - ratio) + GOLD_DARK[2] * ratio)
    d.line(
        [(g_center_x + g_radius - 5 - i//2, g_center_y - g_radius),
         (g_center_x + g_radius - 5 - i//2, g_center_y + g_radius)],
        fill=(r, g, b, 255), width=1
    )

# ===== CANDLESTICKS INSIDE THE G =====
# Draw 3 small candlesticks inside the G loop
candle_data = [
    (g_center_x - 18, g_center_y - 8, 12, 8, True),   # green
    (g_center_x - 5, g_center_y + 2, 10, 12, False),  # red
    (g_center_x + 8, g_center_y - 12, 14, 6, True),   # green
]

for cx, cy, h, w, is_green in candle_data:
    color = GREEN if is_green else RED
    # Wick
    d.line([(cx, cy - h//2), (cx, cy + h//2)], fill=color, width=1)
    # Body
    body_h = h // 2
    d.rectangle([cx - w//2, cy - body_h//2, cx + w//2, cy + body_h//2],
                fill=color, outline=color)

# ===== TEXT: "GAURAVI QUANT" =====
text_x = 140
text_y = H // 2 - 20

# Main text with subtle 3D
for i in range(2, 0, -1):
    d.text((text_x + i, text_y + i), "GAURAVI", font=font_big, fill=(0, 0, 0, 60))

d.text((text_x, text_y), "GAURAVI", font=font_big, fill=WHITE)

# QUANT in gold
quant_y = text_y + 58
for i in range(2, 0, -1):
    d.text((text_x + i, quant_y + i), "QUANT", font=font_small, fill=(0, 0, 0, 60))
d.text((text_x, quant_y), "QUANT", font=font_small, fill=GOLD)

# Tagline
tag_y = quant_y + 24
d.text((text_x, tag_y), "NIFTY 100  ·  Weekly Rankings", font=font_tag, fill=MUTED)

# ===== SUBTLE CHART LINE ACCENT =====
# Draw a mini sparkline under the text
spark_x = text_x
spark_y = tag_y + 22
spark_w = 180
spark_h = 16
points = [0.5, 0.45, 0.52, 0.48, 0.58, 0.55, 0.62, 0.60, 0.65, 0.63, 0.68, 0.66, 0.70]
px = [spark_x + i * (spark_w / (len(points)-1)) for i in range(len(points))]
py = [spark_y + spark_h - p * spark_h for p in points]

# Shadow
for i in range(len(points)-1):
    d.line([(px[i]+1, py[i]+1), (px[i+1]+1, py[i+1]+1)], fill=(0,0,0,50), width=2)
# Main line
for i in range(len(points)-1):
    d.line([(px[i], py[i]), (px[i+1], py[i+1])], fill=GREEN, width=2)
# End dot
d.ellipse([px[-1]-3, py[-1]-3, px[-1]+3, py[-1]+3], fill=GREEN)

# ===== GLOW EFFECT =====
# Create a subtle outer glow
glow = img.filter(ImageFilter.GaussianBlur(radius=8))
glow.putalpha(glow.split()[3].point(lambda x: min(x, 30)))
img = Image.alpha_composite(glow, img)

# Save
out = Path(__file__).parent / "gauravi_logo_3d.png"
img.save(out)
print(f"Logo saved to {out} ({W}x{H})")