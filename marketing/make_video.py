import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG = (13, 17, 23)
PANEL = (22, 27, 34)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
BLUE = (88, 166, 255)
GOLD = (210, 153, 34)
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "marketing"
TMP = Path(tempfile.mkdtemp(prefix="gauravi_video_"))

FONT_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_REG = r"C:\Windows\Fonts\segoeui.ttf"


def font(size, bold=True):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def wrap(draw, text, f, max_w):
    words = text.split()
    lines, line = [], ""
    for w_ in words:
        t = (line + " " + w_).strip()
        if draw.textlength(t, font=f) <= max_w:
            line = t
        else:
            lines.append(line)
            line = w_
    if line:
        lines.append(line)
    return lines


def center_text(draw, y, text, f, fill, lh=None):
    lh = lh or int(f.size * 1.35)
    for ln in wrap(draw, text, f, W - 320):
        tw = draw.textlength(ln, font=f)
        draw.text(((W - tw) / 2, y), ln, font=f, fill=fill)
        y += lh
    return y


def base_slide():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 10], fill=GOLD)
    return img, d


def footer(d, idx):
    f = font(26, False)
    d.text((60, H - 70), "Gauravi Quant  ·  Weekly NIFTY-100 Rankings", font=f, fill=MUTED)
    t = f"{idx} / 6"
    tw = d.textlength(t, font=f)
    d.text((W - 60 - tw, H - 70), t, font=f, fill=MUTED)


def slide_hook():
    img, d = base_slide()
    f_big = font(150)
    tw = d.textlength("90% ACCURACY?", font=f_big)
    d.text(((W - tw) / 2, 300), "90% ACCURACY?", font=f_big, fill=RED)
    fx = font(110)
    tw = d.textlength("❌", font=fx)
    d.text(((W - tw) / 2, 480), "❌", font=fx, fill=TEXT)
    center_text(d, 650, "Nobody with a real edge sells it for ₹999 a month.", font(54), TEXT)
    center_text(d, 740, "So this system was built — and tested honestly.", font(44, False), MUTED)
    footer(d, 1)
    return img


def slide_brand():
    img, d = base_slide()
    y = center_text(d, 250, "Gauravi Quant", font(140), GREEN)
    y = center_text(d, y + 20, "AI-powered weekly ranking system for NIFTY 100", font(52), TEXT)
    boxes = [
        ("TOP 20", "Consider buying", GREEN),
        ("BOTTOM 20", "Avoid or exit", RED),
    ]
    bx = W / 2
    bw, bh = 520, 260
    labels = [(-1, 0), (1, 1)]
    for sx, i in labels:
        x0 = bx + sx * 40 - (bw if sx < 0 else 0)
        x1 = bx - sx * 40 + (bw if sx > 0 else 0)
        title, sub, col = boxes[i]
        d.rounded_rectangle([x0, 600, x1, 600 + bh], radius=24, fill=PANEL, outline=col, width=3)
        ft = font(64)
        tw = d.textlength(title, font=ft)
        d.text(((x0 + x1 - tw) / 2, 650), title, font=ft, fill=col)
        fs = font(36, False)
        tw = d.textlength(sub, font=fs)
        d.text(((x0 + x1 - tw) / 2, 750), sub, font=fs, fill=MUTED)
    center_text(d, 920, "Published every Monday morning", font(38, False), MUTED)
    footer(d, 2)
    return img


def slide_how():
    img, d = base_slide()
    y = center_text(d, 180, "How It Works", font(96), TEXT)
    cards = [
        ("1 · MOMENTUM", "Stocks that beat the market over the last year tend to keep winning.", BLUE),
        ("2 · REVERSAL", "Quality stocks that recently dipped tend to bounce back within weeks.", GOLD),
    ]
    bw, bh = 700, 380
    for i, (title, body, col) in enumerate(cards):
        x0 = W / 2 - bw - 40 if i == 0 else W / 2 + 40
        x1 = x0 + bw
        d.rounded_rectangle([x0, 400, x1, 400 + bh], radius=24, fill=PANEL, outline=col, width=3)
        ft = font(56)
        tw = d.textlength(title, font=ft)
        d.text(((x0 + x1 - tw) / 2, 450), title, font=ft, fill=col)
        fs = font(36, False)
        yy = 550
        for ln in wrap(d, body, fs, bw - 80):
            tw = d.textlength(ln, font=fs)
            d.text(((x0 + x1 - tw) / 2, yy), ln, font=fs, fill=TEXT)
            yy += 52
    center_text(d, 880, "No rumors. No tips. No gut feeling. Just data, ranked, every week.", font(42, False), MUTED)
    footer(d, 3)
    return img


def slide_proof():
    img, d = base_slide()
    y = center_text(d, 160, "Tested, Not Promised", font(96), TEXT)
    stats = [
        ("95,000+", "out-of-sample predictions"),
        ("3 years", "of NIFTY-100 history"),
        ("Walk-forward", "leak-proof validation"),
    ]
    bw, bh = 500, 240
    total = bw * 3 + 80 * 2
    x0 = (W - total) / 2
    for i, (big, small) in enumerate(stats):
        cx = x0 + i * (bw + 80)
        d.rounded_rectangle([cx, 360, cx + bw, 360 + bh], radius=24, fill=PANEL, outline=BLUE, width=3)
        fb = font(72)
        tw = d.textlength(big, font=fb)
        d.text(((cx + bw - tw) / 2, 400), big, font=fb, fill=BLUE)
        fs = font(32, False)
        yy = 510
        for ln in wrap(d, small, fs, bw - 60):
            tw = d.textlength(ln, font=fs)
            d.text(((cx + bw - tw) / 2, yy), ln, font=fs, fill=MUTED)
            yy += 46
    y = center_text(d, 720, "Directional accuracy ≈ 52% — not 90%. We show it upfront.", font(52), RED)
    center_text(d, y + 10, "Small edges + consistency + diversification = how quant funds actually work.", font(38, False), MUTED)
    footer(d, 4)
    return img


def slide_public():
    img, d = base_slide()
    y = center_text(d, 200, "Proof In Public", font(96), TEXT)
    rows = [
        ("Every Monday, 9 AM IST", "a new ranking publishes automatically", GREEN),
        ("Forever archive", "every past report stays online — nothing deleted", BLUE),
        ("Timestamped record", "scroll back and verify any week yourself", GOLD),
    ]
    yy = 420
    for title, sub, col in rows:
        d.rounded_rectangle([360, yy, W - 360, yy + 130], radius=20, fill=PANEL)
        d.rectangle([360, yy, 372, yy + 130], fill=col)
        d.text((410, yy + 18), title, font=font(44), fill=col)
        d.text((410, yy + 74), sub, font=font(32, False), fill=MUTED)
        yy += 160
    footer(d, 5)
    return img


def slide_cta():
    img, d = base_slide()
    y = center_text(d, 280, "Start Free", font(130), GREEN)
    y = center_text(d, y + 30, "Weekly rankings are FREE during public beta.", font(48), TEXT)
    y = center_text(d, y + 10, "🔗 Link in the description", font(56, False), BLUE)
    center_text(d, 800, "Smart investing isn't about hot tips. It's about verified process.", font(42, False), MUTED)
    footer(d, 6)
    return img


SCENES = [
    (slide_hook,
     "Every day you see stock tips promising ninety percent accuracy. Here's the truth. If someone really had that edge, "
     "they would not sell it to you for nine hundred ninety nine rupees a month. So I decided to build my own system, and test it honestly."),
    (slide_brand,
     "This is Gauravi Quant. An A I powered weekly ranking system for the NIFTY one hundred, India's biggest companies. "
     "Every Monday morning, it ranks all one hundred stocks, and publishes two simple lists. The top twenty, to consider buying. "
     "And the bottom twenty, to avoid, or exit."),
    (slide_how,
     "The engine combines two of the most researched effects in stock markets. One, momentum. Stocks that beat the market over the last year "
     "tend to keep winning. Two, reversal. Quality stocks that recently dipped, tend to bounce back over the next few weeks. "
     "No rumors. No tips. No gut feeling. Just data, ranked, every week."),
    (slide_proof,
     "Now here is where Gauravi is different from every telegram channel you have seen. We backtested it properly. Ninety five thousand predictions "
     "across three years, using leak proof walk forward validation, the gold standard in quant finance. And we tell you the uncomfortable number upfront. "
     "Directional accuracy is around fifty two percent. Not ninety. Fifty two. But small edges, applied consistently with diversification, "
     "are exactly how real quantitative funds make money."),
    (slide_public,
     "Every Monday at nine A M, a new ranking is published automatically. And every past report stays online forever. That means when we say "
     "our picks beat the market, you can scroll back, and verify every single week yourself. No deleted messages. No edited calls. "
     "A public, timestamped track record."),
    (slide_cta,
     "Gauravi Quant is in public beta right now. The weekly rankings are completely free, while we build the live track record. "
     "Link is in the description. Subscribe to watch the edge prove itself, week by week, number by number. "
     "Smart investing isn't about hot tips. It's about verified process."),
]


def tts(text, wav_path):
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.SetOutputToWaveFile('%s'); "
        "$s.Rate = -1; "
        "$s.Speak('%s'); "
        "$s.Dispose()"
    ) % (str(wav_path).replace("'", "''"), text.replace("'", "''"))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def main():
    png_dir = OUT_DIR / "slides_png"
    png_dir.mkdir(exist_ok=True)
    wav_dir = TMP / "wav"
    wav_dir.mkdir()

    for i, (render, vo) in enumerate(SCENES, 1):
        png = png_dir / f"scene_{i}.png"
        render().save(png)
        wav = wav_dir / f"scene_{i}.wav"
        print(f"[{i}/6] rendering slide + voiceover ...")
        tts(vo, wav)
        clip = TMP / f"scene_{i}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-loop", "1", "-framerate", "30", "-i", str(png),
            "-i", str(wav),
            "-filter_complex", "[1:a]apad=pad_dur=1.0[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(clip),
        ], check=True)

    lst = TMP / "list.txt"
    lst.write_text("".join(f"file 'scene_{i}.mp4'\n" for i in range(1, 7)))
    final = OUT_DIR / "gauravi_quant_youtube.mp4"
    print("concatenating final video ...")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", str(final),
    ], check=True)
    print(f"DONE -> {final}")


if __name__ == "__main__":
    sys.exit(main())
