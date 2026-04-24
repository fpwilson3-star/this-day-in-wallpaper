#!/usr/bin/env python3
"""Generate a daily AMOLED wallpaper based on the most interesting science/tech/medicine fact."""

import datetime
import json
import os
import re
import textwrap
from io import BytesIO
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PHONE_WIDTH = 1080
PHONE_HEIGHT = 2340
IMAGE_MODEL = "gemini-3-pro-image-preview"
TEXT_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Step 1 — pick today's fact and build an image prompt
# ---------------------------------------------------------------------------

FACT_SYSTEM = """\
You are a science communicator who selects the single most fascinating fact, discovery, or \
milestone from science, medicine, or technology for a given date. You prioritize things that \
are visually compelling and mind-blowing. You always respond with valid JSON only."""

FACT_USER = """\
Today is {date_long}.

Choose the MOST interesting science, medicine, or technology fact or event associated with \
this calendar date (any year) OR something genuinely remarkable happening in the world right now. \
Use your judgment — pick the one a curious person would most want to see on their phone screen \
every morning.

Return a JSON object with exactly these keys:
  "fact"         : 1–2 sentence human-readable description of the fact (shown as wallpaper caption)
  "image_prompt" : Detailed Imagen prompt for a portrait phone wallpaper.
                   Rules for the prompt:
                   - Pure black (#000000) background — this is non-negotiable
                   - Style: Scientific infographic, schematic, white on black
                   - Visually represent the specific fact with recognisable scientific imagery
                   - No text, no words, no labels in the image
                   - Portrait orientation, highly detailed, cinematic"""


def get_fact_and_prompt(client: genai.Client, date: datetime.date) -> tuple[str, str]:
    date_long = date.strftime("%B %d")
    prompt = FACT_USER.format(date_long=date_long)

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=FACT_SYSTEM,
            response_mime_type="application/json",
            temperature=1.0,
        ),
    )

    raw = response.text.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)
    return data["fact"], data["image_prompt"]


# ---------------------------------------------------------------------------
# Step 2 — generate the image with Gemini
# ---------------------------------------------------------------------------


def generate_image(client: genai.Client, prompt: str, retries: int = 3) -> Image.Image:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=IMAGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    img = Image.open(BytesIO(part.inline_data.data)).convert("RGB")
                    return img
            raise RuntimeError("No image returned in Gemini response")
        except Exception as exc:
            last_exc = exc
            print(f"Image generation attempt {attempt} failed: {exc}")
            if attempt < retries:
                import time
                time.sleep(10 * attempt)
    raise last_exc


# ---------------------------------------------------------------------------
# Step 3 — fit image to phone canvas (pure black padding)
# ---------------------------------------------------------------------------


def fit_to_phone(img: Image.Image) -> Image.Image:
    """Scale img to fill width, then center-crop/pad to phone height."""
    target_w, target_h = PHONE_WIDTH, PHONE_HEIGHT

    # Scale to fill width
    scale = target_w / img.width
    new_h = int(img.height * scale)
    img = img.resize((target_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    y_offset = (target_h - new_h) // 2  # negative → crop; positive → pad
    canvas.paste(img, (0, y_offset))
    return canvas


# ---------------------------------------------------------------------------
# Step 4 — overlay fact text at the bottom
# ---------------------------------------------------------------------------

FONT_PATH_CANDIDATES = [
    # Ubuntu (GitHub Actions runner)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    # Windows
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATH_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def add_text_overlay(img: Image.Image, fact: str, date: datetime.date) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font_fact = _load_font(38)
    font_date = _load_font(28)

    # Wrap fact text to fit within margins
    margin = 60
    max_chars = 42  # approximate chars per line at font size 38
    lines = textwrap.wrap(fact, width=max_chars)

    line_height = 48
    date_str = date.strftime("%B %d, %Y")
    text_block_height = len(lines) * line_height + 50  # 50 for date line + gap

    # Gradient overlay: bottom portion fades from transparent to black
    gradient_h = text_block_height + 120
    gradient = Image.new("RGBA", (w, gradient_h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(gradient)
    for i in range(gradient_h):
        alpha = int(220 * (i / gradient_h))
        gd.line([(0, i), (w, i)], fill=(0, 0, 0, alpha))

    img_rgba = img.convert("RGBA")
    img_rgba.paste(gradient, (0, h - gradient_h), gradient)
    img = img_rgba.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw date line
    y = h - text_block_height - 10
    date_color = (120, 200, 255)  # pale cyan
    draw.text((margin, y), date_str, font=font_date, fill=date_color)

    # Draw fact lines with soft glow effect
    y += 40
    glow_color = (80, 80, 80)
    text_color = (240, 240, 240)
    for line in lines:
        # Glow pass (blurred shadow)
        tmp = Image.new("RGBA", img.size, (0, 0, 0, 0))
        tdraw = ImageDraw.Draw(tmp)
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2), (2, 0)]:
            tdraw.text((margin + dx, y + dy), line, font=font_fact, fill=(*glow_color, 180))
        glow = tmp.filter(ImageFilter.GaussianBlur(2))
        img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.text((margin, y), line, font=font_fact, fill=text_color)
        y += line_height

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)

    today = datetime.date.today()
    date_str = today.strftime("%Y-%m-%d")

    print(f"Generating wallpaper for {date_str} ...")

    fact, image_prompt = get_fact_and_prompt(client, today)
    print(f"\nFact: {fact}")
    print(f"\nImage prompt:\n{image_prompt}\n")

    raw_img = generate_image(client, image_prompt)
    fitted = fit_to_phone(raw_img)
    final = add_text_overlay(fitted, fact, today)

    out_dir = Path("wallpaper")
    out_dir.mkdir(exist_ok=True)

    dated_path = out_dir / f"{date_str}.png"
    latest_path = out_dir / "latest.png"
    fact_path = out_dir / "latest_fact.txt"

    final.save(dated_path, optimize=True)
    final.save(latest_path, optimize=True)
    fact_path.write_text(f"{date_str}\n{fact}\n", encoding="utf-8")

    print(f"Saved: {dated_path}")
    print(f"Saved: {latest_path}")


if __name__ == "__main__":
    main()
