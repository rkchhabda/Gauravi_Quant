from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path

FONT_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_REG = r"C:\Windows\Fonts\segoeui.ttf"

GOLD = (210, 153, 34, 255)
GREEN = (63, 185, 80, 255)
RED = (248, 81, 73, 255)
WHITE = (230, 237, 243, 255)
MUTED = (139, 148, 158, 200)
BG_PANEL = (22, 27, 34, 180)

def text_width(text, font):
    # Use a dummy draw to measure
    dummy = Image.new("RGBA", (10, 10))
    d = ImageDraw.Draw(dummy)
    return d.textlength(text, font=font)

# Design 1: Minimal Horizontal - Fixed spacing
def design_1():
    W, H = 360, 80
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    
    font_big = ImageFont.truetype(FONT_BOLD, 48)
    font_small = ImageFont.truetype(FONT_REG, 16)
    
    # Measure text widths
    gauravi_w = text_width("GAURAVI", font_big)
    quant_w = text_width("QUANT", font_small)
    
    # Candlesticks on far left
    cx, cy = 20, H // 2
    for i, (h, green) in enumerate([(18, True), (14, False), (20, True)]):
        x = cx + i * 12
        color = GREEN if green else RED
        d.line([(x, cy - h//2), (x, cy + h//2)], fill=color, width=2)
        d.rectangle([x-3, cy-4, x+3, cy+4], fill=color)
    
    # Text starts after candlesticks with gap
    tx = cx + 3*12 + 16  # ~52 + 16 = 68
    d.text((tx, 10), "GAURAVI", font=font_big, fill=WHITE)
    
    # QUANT positioned after GAURAVI with comfortable gap
    quant_x = tx + gauravi_w + 20
    d.text((quant_x, 28), "QUANT", font=font_small, fill=GOLD)
    
    # Tagline below GAURAVI
    d.text((tx, 54), "NIFTY 100 Rankings", font=font_small, fill=MUTED)
    
    img.save(Path(__file__).parent / "logo_option_1.png")
    print(f"Option 1: GAURAVI width={gauravi_w:.0f}, QUANT at x={quant_x:.0f}")

# Design 2: Compact Square Badge - Fixed
def design_2():
    W, H = 160, 160
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    
    GOLD_DARK = (160, 110, 20, 255)
    font_big = ImageFont.truetype(FONT_BOLD, 36)
    font_small = ImageFont.truetype(FONT_BOLD, 12)
    
    center = W // 2
    r = 55
    
    # Circular background
    for i in range(r, 0, -1):
        alpha = int(30 * (1 - i/r))
        d.ellipse([center-i, center-i, center+i, center+i], fill=(13, 17, 23, alpha))
    
    # 3D "G" with candlestick inside
    for i in range(4, 0, -1):
        d.arc([center-r+i, center-r+i, center+r+i, center+r+i], 220, 140, fill=(0,0,0,40), width=6)
    
    for i in range(6):
        ratio = i/6
        col = tuple(int(GOLD[c]*(1-ratio) + GOLD_DARK[c]*ratio) for c in range(3)) + (255,)
        d.arc([center-r+i//2, center-r+i//2, center+r-i//2, center+r-i//2], 220, 140, fill=col, width=1)
    
    for i in range(6):
        ratio = i/6
        col = tuple(int(GOLD[c]*(1-ratio) + GOLD_DARK[c]*ratio) for c in range(3)) + (255,)
        d.line([(center-5, center+i//2), (center+r-8, center+i//2)], fill=col, width=1)
    
    for i in range(6):
        ratio = i/6
        col = tuple(int(GOLD[c]*(1-ratio) + GOLD_DARK[c]*ratio) for c in range(3)) + (255,)
        d.line([(center+r-8-i//2, center-r), (center+r-8-i//2, center+r)], fill=col, width=1)
    
    # Tiny candlesticks inside
    for dx, dy, h, green in [(-12, -6, 16, True), (0, 2, 12, False), (10, -8, 18, True)]:
        x, y = center + dx, center + dy
        color = GREEN if green else RED
        d.line([(x, y-h//2), (x, y+h//2)], fill=color, width=1)
        d.rectangle([x-2, y-3, x+2, y+3], fill=color)
    
    # "GAURAVI QUANT" centered below
    text = "GAURAVI QUANT"
    tw = text_width(text, font_small)
    d.text((center - tw//2, 130), text, font=font_small, fill=WHITE)
    
    img.save(Path(__file__).parent / "logo_option_2.png")
    print("Option 2: Compact square badge (160x160)")

# Design 3: Wide Banner with Panel - Fixed
def design_3():
    W, H = 400, 70
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    
    font_big = ImageFont.truetype(FONT_BOLD, 40)
    font_small = ImageFont.truetype(FONT_REG, 14)
    
    # Subtle panel background
    d.rounded_rectangle([0, 0, W, H], radius=8, fill=BG_PANEL)
    
    # Mini sparkline on left
    sx, sy = 16, H//2
    sw, sh = 60, 30
    pts = [0.45, 0.42, 0.5, 0.48, 0.55, 0.52, 0.6, 0.58, 0.65, 0.63, 0.68]
    px = [sx + i * sw/(len(pts)-1) for i in range(len(pts))]
    py = [sy + sh/2 - p*sh for p in pts]
    for i in range(len(pts)-1):
        d.line([(px[i], py[i]), (px[i+1], py[i+1])], fill=GREEN, width=2)
    d.ellipse([px[-1]-3, py[-1]-3, px[-1]+3, py[-1]+3], fill=GOLD)
    
    # Text with measured spacing
    tx = 90
    gauravi_w = text_width("GAURAVI", font_big)
    d.text((tx, 6), "GAURAVI", font=font_big, fill=WHITE)
    
    quant_x = tx + gauravi_w + 12
    d.text((quant_x, 6), "QUANT", font=font_big, fill=GOLD)
    
    d.text((tx, 42), "Weekly NIFTY-100 Rankings", font=font_small, fill=MUTED)
    
    # Tiny green dot indicator at far right
    d.ellipse([W-30, 20, W-18, 32], fill=GREEN)
    
    img.save(Path(__file__).parent / "logo_option_3.png")
    print(f"Option 3: GAURAVI width={gauravi_w:.0f}, QUANT at x={quant_x:.0f}")

# Design 4: Ultra-minimal with Accent Bar - Fixed
def design_4():
    W, H = 300, 60
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    
    font_big = ImageFont.truetype(FONT_BOLD, 42)
    font_small = ImageFont.truetype(FONT_REG, 14)
    
    # Gold accent bar
    d.rectangle([0, 0, 4, H], fill=GOLD)
    
    # Measure and position properly
    gauravi_w = text_width("GAURAVI", font_big)
    quant_w = text_width("QUANT", font_big)
    
    tx = 16
    d.text((tx, 6), "GAURAVI", font=font_big, fill=WHITE)
    
    quant_x = tx + gauravi_w + 16
    d.text((quant_x, 6), "QUANT", font=font_big, fill=GOLD)
    
    d.text((tx, 38), "NIFTY-100  ·  Weekly Rankings", font=font_small, fill=MUTED)
    
    # Tiny candlestick triplet at far right with gap from QUANT
    candle_start = quant_x + quant_w + 20
    for i, (h, green) in enumerate([(16, True), (12, False), (18, True)]):
        x = candle_start + i * 12
        color = GREEN if green else RED
        d.line([(x, H//2 - h//2), (x, H//2 + h//2)], fill=color, width=2)
        d.rectangle([x-2, H//2-3, x+2, H//2+3], fill=color)
    
    img.save(Path(__file__).parent / "logo_option_4.png")
    print(f"Option 4: GAURAVI={gauravi_w:.0f}, QUANT at={quant_x:.0f}, candles at={candle_start:.0f}")

# Run all
design_1()
design_2()
design_3()
design_4()
print("\nAll 4 options regenerated with fixed text spacing.")