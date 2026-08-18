import os
import re
import json
import glob
import html
import base64
from io import BytesIO
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import yt_dlp
from moviepy import VideoFileClip, CompositeVideoClip, ImageClip

# Configuració
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
INSTAGRAM_COOKIES_FILE = os.getenv("INSTAGRAM_COOKIES_FILE")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEST_MODE = False


# ==========================================
# GESTIÓ D'HISTORIAL
# ==========================================

def load_processed_ids():
    if os.path.exists("processed_videos.json"):
        try:
            with open("processed_videos.json", "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_processed_id(video_id):
    if TEST_MODE:
        return
    history = load_processed_ids()
    if video_id not in history:
        history.append(video_id)
        with open("processed_videos.json", "w") as f:
            json.dump(history, f, indent=4)


# ==========================================
# UTILITATS D'IMATGE I VISIÓ AI
# ==========================================

def extract_frame_as_image(video_path, timestamp=0.5):
    """Extreu un fotograma del vídeo com a objecte PIL Image."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame = cap.read()
    cap.release()
    
    if success and frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None


def image_to_base64_jpeg(image_pil):
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def analyze_with_groq_vision(image_pil, caption_raw=""):
    """Analitza la imatge amb els models de visió actius de Groq."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    prompt = (
        "Examine this video frame and the original post description.\n"
        "1. OCR / Read the header text, tweet, or meme text visible in the image.\n"
        "2. Identify the original author/account handle (e.g. @username from the avatar/watermark or caption). If none, set 'Unknown'.\n"
        "3. PARAPHRASE the text into a clean, viral, natural tweet/post text in ENGLISH (1-3 short punchy sentences). Keep the exact same joke, tone, and context.\n"
        "4. Generate an engaging Instagram/TikTok caption in English with a Call to Action and 4-6 viral hashtags.\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        '{"credits": "@author", "tweet_text": "Translated and paraphrased tweet text...", "generated_caption": "Caption with #hashtags..."}'
    )
    
    if caption_raw:
        prompt += f"\nOriginal caption: {caption_raw}"

    content = [{"type": "text", "text": prompt}]
    if image_pil is not None:
        b64 = image_to_base64_jpeg(image_pil)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    # Models de visió oficials de Groq
    candidate_models = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3.6-27b"
    ]

    for model_name in candidate_models:
        try:
            print(f"🧠 Analitzant amb Groq Vision ({model_name})...")
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.6
            )
            data = json.loads(completion.choices[0].message.content)
            return (
                data.get("credits", "Unknown"),
                data.get("tweet_text", ""),
                data.get("generated_caption", "")
            )
        except Exception as e:
            print(f"ℹ️ Groq error ({model_name}): {e}")
            continue
    return None


def analyze_with_gemini_vision(image_pil, caption_raw=""):
    """Fallback amb Google Gemini (gemini-3.6-flash / gemini-2.0-flash)."""
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = """
    Examine this video frame and caption.
    1. Read the header text/tweet in the image.
    2. Identify original creator handle (@user).
    3. Paraphrase the text into natural English tweet style (1-3 sentences).
    4. Generate viral Instagram caption with hashtags.
    Return JSON: {"credits": "@author", "tweet_text": "...", "generated_caption": "..."}
    """
    
    contents = [prompt]
    if image_pil is not None:
        contents.append(image_pil)
    if caption_raw:
        contents.append(f"\nOriginal video description: {caption_raw}")

    candidate_models = ["gemini-3.6-flash", "gemini-2.0-flash"]
    for model_name in candidate_models:
        try:
            print(f"🧠 Analitzant amb Gemini ({model_name})...")
            res = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(res.text)
            return (
                data.get("credits", "Unknown"),
                data.get("tweet_text", ""),
                data.get("generated_caption", "")
            )
        except Exception as e:
            print(f"ℹ️ Gemini error ({model_name}): {e}")
            continue
    return None


def analyze_content(image_pil, caption_raw=""):
    """Prova primer Groq; si falla, prova Gemini; si falla, retorna valors per defecte."""
    if GROQ_API_KEY:
        res = analyze_with_groq_vision(image_pil, caption_raw)
        if res:
            return res

    if GEMINI_API_KEY:
        res = analyze_with_gemini_vision(image_pil, caption_raw)
        if res:
            return res

    print("⚠️ No s'ha pogut utilitzar cap API de Visió. S'utilitzen valors per defecte.")
    return "Unknown", "Check this out!", caption_raw


# ==========================================
# RENDERITZAT ESTIL TWEET (PILLOW)
# ==========================================

def get_system_font(font_name="bold", size=32):
    """Busca una font moderna (Montserrat o DejaVu) al sistema."""
    candidates = []
    if font_name == "bold":
        candidates = [
            "/usr/share/fonts/truetype/montserrat/Montserrat-Bold.ttf",
            "/usr/share/fonts/truetype/montserrat/Montserrat-SemiBold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "Arial Bold.ttf"
        ]
    elif font_name == "regular":
        candidates = [
            "/usr/share/fonts/truetype/montserrat/Montserrat-Medium.ttf",
            "/usr/share/fonts/truetype/montserrat/Montserrat-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "Arial.ttf"
        ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap_text(text, font, max_width, draw):
    """Ajusta les línies de text perquè no superin l'amplada de la pantalla."""
    lines = []
    paragraphs = text.split("\n")
    for para in paragraphs:
        if not para.strip():
            lines.append("")
            continue
        words = para.split()
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
            if w <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
        if current_line:
            lines.append(" ".join(current_line))
    return lines


def create_tweet_header_image(tweet_text, width=1080):
    """Genera la capçalera estil Tweet amb Avatar, Feedity, @feedity.tv i el text."""
    margin_x = 70
    max_text_width = width - (margin_x * 2)

    name_font = get_system_font("bold", size=40)
    handle_font = get_system_font("regular", size=34)
    body_font = get_system_font("bold", size=46)

    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    wrapped_lines = wrap_text(tweet_text, body_font, max_text_width, dummy_draw)

    line_height = 64
    body_height = len(wrapped_lines) * line_height
    avatar_size = 90
    top_padding = 60
    bottom_padding = 45
    header_height = top_padding + avatar_size + 30 + body_height + bottom_padding

    img = Image.new("RGBA", (width, header_height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    # 1. Avatar Circular
    avatar_x = margin_x
    avatar_y = top_padding
    if os.path.exists("logo.png"):
        try:
            logo = Image.open("logo.png").convert("RGBA").resize((avatar_size, avatar_size))
            mask = Image.new("L", (avatar_size, avatar_size), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, avatar_size, avatar_size), fill=255)
            img.paste(logo, (avatar_x, avatar_y), mask)
        except Exception:
            pass
    else:
        draw.ellipse([avatar_x, avatar_y, avatar_x + avatar_size, avatar_y + avatar_size], fill=(24, 24, 24))
        draw.text((avatar_x + 28, avatar_y + 16), "F", font=name_font, fill=(245, 200, 30))

    # 2. Nom i Username
    text_start_x = avatar_x + avatar_size + 24
    draw.text((text_start_x, avatar_y + 4), "Feedity", font=name_font, fill=(255, 255, 255))
    draw.text((text_start_x, avatar_y + 48), "@feedity.tv", font=handle_font, fill=(120, 130, 140))

    # 3. Cos del Tweet
    text_y = avatar_y + avatar_size + 30
    for line in wrapped_lines:
        draw.text((margin_x, text_y), line, font=body_font, fill=(255, 255, 255))
        text_y += line_height

    return np.array(img)


# ==========================================
# DETECCIÓ I CROP INTEL·LIGENT DE CONTINGUT
# ==========================================

def detect_background_color(frame):
    h, w, _ = frame.shape
    top_sample = frame[0:15, w//4:3*w//4]
    bottom_sample = frame[h-15:h, w//4:3*w//4]
    samples = np.concatenate([top_sample, bottom_sample], axis=0)
    return np.median(samples, axis=(0, 1))


def get_longest_consecutive_run(bool_array):
    max_run = 0
    current_run = 0
    for val in bool_array:
        if val:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return max_run


def find_video_box_in_frame(frame, color_diff_threshold=30, min_width_ratio=0.65, min_height_ratio=0.65):
    h, w, _ = frame.shape
    bg_color = detect_background_color(frame)

    diff = np.max(np.abs(frame.astype(np.float32) - bg_color), axis=2)
    foreground_mask = diff > color_diff_threshold

    min_continuous_px = int(w * min_width_ratio)
    valid_y = []

    for y in range(h):
        row = foreground_mask[y, :]
        if get_longest_consecutive_run(row) >= min_continuous_px:
            valid_y.append(y)

    if not valid_y:
        return None

    y1 = valid_y[0]
    y2 = valid_y[-1]

    if (y2 - y1) < 100:
        return None

    video_region_mask = foreground_mask[y1:y2, :]
    region_h = y2 - y1
    min_col_pixels = int(region_h * min_height_ratio)

    valid_x = []
    for x in range(w):
        col = video_region_mask[:, x]
        if np.sum(col) >= min_col_pixels:
            valid_x.append(x)

    if not valid_x:
        x1, x2 = 0, w
    else:
        x1 = valid_x[0]
        x2 = valid_x[-1]

    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def crop_content_bounding_box(clip, num_samples=6):
    duration = clip.duration
    if not duration or duration <= 0:
        return None

    timestamps = np.linspace(0.5, max(duration - 0.5, 0.5), num=num_samples)
    boxes = []

    for t in timestamps:
        try:
            frame = clip.get_frame(t)
            box = find_video_box_in_frame(frame)
            if box:
                boxes.append(box)
        except Exception:
            continue

    if not boxes:
        return None

    boxes = np.array(boxes)
    median_box = np.median(boxes, axis=0).astype(int)
    x, y, w, h = median_box
    return (int(x), int(y), int(w), int(h))


# ==========================================
# RENDERITZAT FINAL AMB FORMAT TWEET
# ==========================================

def process_video_canvas(input_path, tweet_text, output_path="final_feedity.mp4"):
    clip = VideoFileClip(input_path)
    frame_w, frame_h = clip.w, clip.h
    bbox = crop_content_bounding_box(clip)

    min_area_ratio = 0.10
    if bbox and (bbox[2] * bbox[3]) >= min_area_ratio * frame_w * frame_h:
        x, y, w, h = bbox
        margin_x = int(w * 0.01)
        margin_y = int(h * 0.01)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame_w, x + w + margin_x)
        y2 = min(frame_h, y + h + margin_y)
        print(f"✂️ Crop aplicat amb èxit: x={x1}, y={y1}, x2={x2}, y2={y2}")
        cropped_clip = clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
    else:
        print("ℹ️ No s'ha detectat cap marc distintiu, s'utilitza el vídeo original.")
        cropped_clip = clip

    scaled_clip = cropped_clip.resized(width=1080)

    # Generem la capçalera estil Tweet
    print("🎨 Renderitzant capçalera estil Tweet...")
    header_img_np = create_tweet_header_image(tweet_text, width=1080)
    header_h = header_img_np.shape[0]

    header_clip = ImageClip(header_img_np).with_duration(scaled_clip.duration)

    # Posicionament
    header_clip = header_clip.with_position(("center", 180))
    video_y_pos = 180 + header_h + 10
    video_positioned = scaled_clip.with_position(("center", video_y_pos))

    final_clip = CompositeVideoClip([video_positioned, header_clip], size=(1080, 1920))

    final_clip.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac",
        fps=30,
        preset="fast"
    )

    clip.close()
    cropped_clip.close()
    final_clip.close()
    header_clip.close()


# ==========================================
# NOTIFICACIÓ TELEGRAM
# ==========================================

def send_telegram_notification(video_path, tweet_text, credits, generated_caption, video_id):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Notificació de Telegram omessa (tokens no configurats).")
        return

    safe_tweet = html.escape(tweet_text or "")
    safe_credits = html.escape(credits or "Unknown")
    safe_video_id = html.escape(video_id or "")

    max_caption_len = 650
    if len(generated_caption) > max_caption_len:
        generated_caption = generated_caption[:max_caption_len] + "..."
    safe_caption = html.escape(generated_caption)

    message_html = (
        f"🎬 <b>NOU VÍDEO PROCESSAT PER A FEEDITY</b>\n\n"
        f"📌 <b>Tweet Text</b>:\n<i>{safe_tweet}</i>\n\n"
        f"👤 <b>Crèdits Originals</b>: {safe_credits}\n"
        f"🆔 <b>ID</b>: <code>{safe_video_id}</code>\n\n"
        f"📝 <b>CAPTION PER PUBLICAR</b>:\n{safe_caption}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    with open(video_path, "rb") as video_file:
        files = {"video": video_file}
        data = {
            "chat_id": TELEGRAM_CHAT_ID, 
            "caption": message_html,
            "parse_mode": "HTML"
        }
        res = requests.post(url, data=data, files=files)
        if res.status_code == 200:
            print("🚀 Vídeo i dades enviats correctament a Telegram!")
        else:
            print(f"❌ Error en enviar a Telegram: {res.text}")


# ==========================================
# FLUX PRINCIPAL
# ==========================================

def extract_shortcode(reel_url):
    match = re.search(r"instagram\.com/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)", reel_url)
    return match.group(1) if match else None


def _cleanup_temp_input():
    for f in glob.glob("temp_input.*"):
        try:
            os.remove(f)
        except OSError:
            pass


def get_reel_by_url(reel_url):
    processed_ids = load_processed_ids()
    shortcode = extract_shortcode(reel_url)

    if shortcode and shortcode in processed_ids:
        print(f"⏭️ Reel ja processat anteriorment ({shortcode}), s'omet.")
        return None, None, None, None, None

    print(f"⬇️ Descarregant reel amb yt-dlp: {reel_url}")
    _cleanup_temp_input()

    ydl_opts = {
        "outtmpl": "temp_input.%(ext)s",
        "format": "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if INSTAGRAM_COOKIES_FILE and os.path.exists(INSTAGRAM_COOKIES_FILE):
        ydl_opts["cookiefile"] = INSTAGRAM_COOKIES_FILE

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(reel_url, download=True)
    except yt_dlp.utils.DownloadError as e:
        print(f"❌ Error en descarregar el reel amb yt-dlp: {e}")
        return None, None, None, None, None
    except Exception as e:
        print(f"❌ Error inesperat amb yt-dlp: {e}")
        return None, None, None, None, None

    video_id = str(info.get("id") or shortcode or reel_url)
    if video_id in processed_ids:
        print(f"⏭️ Reel ja processat anteriorment ({video_id}), s'omet.")
        return None, None, None, None, None

    downloaded_path = ydl.prepare_filename(info)
    if not os.path.exists(downloaded_path):
        candidates = glob.glob("temp_input.*")
        downloaded_path = candidates[0] if candidates else None

    if not downloaded_path or not os.path.exists(downloaded_path):
        print("⚠️ yt-dlp no ha generat cap fitxer de vídeo.")
        return None, None, None, None, None

    caption_raw = info.get("description") or ""

    print("🤖 Analitzant contingut visual i text del reel...")
    frame_image = extract_frame_as_image(downloaded_path, timestamp=0.5)
    credits, tweet_text, generated_caption = analyze_content(frame_image, caption_raw)

    return downloaded_path, video_id, credits, tweet_text, generated_caption


def main():
    if not os.path.exists("sources.csv"):
        print("❌ No s'ha trobat el fitxer sources.csv")
        return

    with open("sources.csv", "r") as f:
        reel_urls = [
            line.strip() for line in f
            if line.strip() and not line.startswith("reel_url") and not line.startswith("account_handle")
        ]

    for reel_url in reel_urls:
        print(f"\n🚀 Processant reel: {reel_url}")
        video_file, video_id, credits, tweet_text, generated_caption = get_reel_by_url(reel_url)

        if video_file:
            process_video_canvas(video_file, tweet_text, "final_feedity.mp4")
            send_telegram_notification("final_feedity.mp4", tweet_text, credits, generated_caption, video_id)
            save_processed_id(video_id)
            print("✅ Procés finalitzat amb èxit!")
            break
        else:
            print(f"⚠️ Reel omès o sense contingut nou: {reel_url}")


if __name__ == "__main__":
    main()
