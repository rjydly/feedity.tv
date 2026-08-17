import os
import re
import json
import glob
import urllib.request
import requests
import numpy as np
import cv2
import ollama
import yt_dlp
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, CompositeVideoClip, ImageClip, ColorClip

# =============================================================================
# CONFIGURACIÓ DE DISSENY I FONS
# =============================================================================
# Color del lienzo (RGB): Verd debug
CANVAS_BG_COLOR = (0, 255, 0)

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
LOGO_PATH = "assets/logo.png"

# Descàrrega temporal de la font Lexend des de Google Fonts al directori /tmp/
FONT_DIR = "/tmp/fonts"
LEXEND_REGULAR_PATH = os.path.join(FONT_DIR, "Lexend-Regular.ttf")
LEXEND_BOLD_PATH = os.path.join(FONT_DIR, "Lexend-Bold.ttf")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
INSTAGRAM_COOKIES_FILE = os.getenv("INSTAGRAM_COOKIES_FILE")
TEST_MODE = False


def setup_fonts():
    """Baixa les fonts Lexend directament de Google Fonts si no estan a /tmp/."""
    os.makedirs(FONT_DIR, exist_ok=True)
    urls = {
        LEXEND_REGULAR_PATH: "https://github.com/google/fonts/raw/main/ofl/lexend/Lexend%5Bwght%5D.ttf",
        LEXEND_BOLD_PATH: "https://github.com/google/fonts/raw/main/ofl/lexend/Lexend%5Bwght%5D.ttf"
    }
    for path, url in urls.items():
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"⚠️ Error descarregant la font ({url}): {e}", flush=True)


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


def analyze_caption_with_local_ai(caption):
    """
    Prompt millorat amb exemples (Few-Shot Prompting) per obtenir títols
    virals, moderns, amb slang de xarxes i etiquetes de negreta ben aplicades.
    """
    prompt = f"""
    You are the lead social media copywriter for @feedity.tv, a viral media brand.
    Analyze the raw video caption below and generate a modern, engaging headline and Instagram post.

    Raw caption: "{caption}"

    STRICT INSTRUCTIONS:
    1. **LANGUAGE**: ALL generated content MUST be written strictly in ENGLISH.
    2. **credits**: Extract the original creator's handle (e.g. @creator). Return "Unknown" if not found.
    3. **headline**: Create a 10 to 16-word VIRAL HEADLINE for the top video header.
       - Style: Use current brainrot/internet slang, clickbait, or relatable hooks (e.g. "bro thought", "no way he actually", "living rent free", "insane plot twist", "nah this is crazy").
       - Format: Wrap the MOST IMPORTANT words in <b> and </b> tags so they get rendered in BOLD.
       - NEVER leave it blank and NEVER output generic titles like "Viral moment".
    4. **generated_caption**: Write a short, engaging Instagram caption with emojis and trending hashtags. Use ONLY @feedity.tv for CTAs.

    EXAMPLES OF GOOD HEADLINES:
    - "Bro really thought he could <b>get away with this</b> live on national TV 💀"
    - "Nah this <b>plot twist</b> caught everyone completely off guard 😭"
    - "This core memory will live <b>rent free in my head</b> forever"
    - "She literally unlocked a <b>hidden side quest</b> during the audition 💀"

    Respond ONLY with a valid JSON object:
    {{
      "credits": "@username",
      "headline": "Bro really thought he could <b>get away with this</b> 💀",
      "generated_caption": "Wait for the end! 🍿 Follow @feedity.tv for more viral clips!\n\n#viral #funny"
    }}
    """
    try:
        response = ollama.chat(
            model='gemma2',
            messages=[{'role': 'user', 'content': prompt}]
        )
        content = response['message']['content']
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            credits = data.get("credits", "Unknown")
            headline = data.get("headline", "").strip()
            gen_caption = data.get("generated_caption", "").strip()
            
            if not headline or "feedity" in headline.lower():
                headline = "Nah this <b>unbelievable moment</b> caught everyone off guard 💀"

            gen_caption = re.sub(r'@[A-Za-z0-9_.]+', '@feedity.tv', gen_caption)
            
            if credits and credits != "Unknown":
                gen_caption = f"{gen_caption}\n\nVia: {credits}"
                
            return credits, headline, gen_caption
    except Exception as e:
        print(f"⚠️ Error analitzant amb Ollama: {e}", flush=True)
    
    return "Unknown", "Bro really thought he could <b>pull this off</b> live on camera 💀", f"Follow @feedity.tv for more! 🍿\n\n{caption}"

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
        print(f"⏭️ Reel ja processat anteriorment ({shortcode}), s'omet.", flush=True)
        return None, None, None, None, None

    print(f"⬇️ Descarregant reel amb yt-dlp: {reel_url}", flush=True)
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
    except Exception as e:
        print(f"❌ Error descarregant amb yt-dlp: {e}", flush=True)
        return None, None, None, None, None

    video_id = str(info.get("id") or shortcode or reel_url)
    if video_id in processed_ids:
        print(f"⏭️ Reel ja processat ({video_id}), s'omet.", flush=True)
        return None, None, None, None, None

    downloaded_path = ydl.prepare_filename(info)
    if not os.path.exists(downloaded_path):
        candidates = glob.glob("temp_input.*")
        downloaded_path = candidates[0] if candidates else None

    if not downloaded_path or not os.path.exists(downloaded_path):
        print("❌ No s'ha pogut localitzar el fitxer descarregat.", flush=True)
        return None, None, None, None, None

    caption_raw = info.get("description") or ""

    print("🤖 Analitzant contingut amb Gemma 2...", flush=True)
    credits, headline, generated_caption = analyze_caption_with_local_ai(caption_raw)

    return downloaded_path, video_id, credits, headline, generated_caption


def _sample_frames_grayscale_from_clip(clip, num_samples=10):
    duration = clip.duration
    if not duration or duration <= 0:
        return []
    safe_end = max(duration - 0.05, 0)
    timestamps = np.linspace(0, safe_end, num=num_samples)
    frames = []
    for t in timestamps:
        try:
            frame = clip.get_frame(t)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32))
        except Exception:
            continue
    return frames


def _refine_micro_borders(frame_crop, threshold_white=215, max_check_pixels=15):
    """Fase 2: Retalla micro-bordes residuals clars."""
    h, w = frame_crop.shape
    top, bottom, left, right = 0, 0, 0, 0

    for y in range(min(max_check_pixels, h)):
        if np.mean(frame_crop[y, :]) > threshold_white:
            top = y + 1
        else:
            break

    for y in range(1, min(max_check_pixels, h)):
        if np.mean(frame_crop[h - y, :]) > threshold_white:
            bottom = y
        else:
            break

    for x in range(min(max_check_pixels, w)):
        if np.mean(frame_crop[:, x]) > threshold_white:
            left = x + 1
        else:
            break

    for x in range(1, min(max_check_pixels, w)):
        if np.mean(frame_crop[:, w - x]) > threshold_white:
            right = x
        else:
            break

    return top, bottom, left, right


def crop_content_bounding_box(clip):
    """Detecció i retall de fons en 2 FASES (Global + Micro-crop)."""
    frames = _sample_frames_grayscale_from_clip(clip)
    if not frames:
        return None

    stacked = np.stack(frames, axis=0)
    mean_frame = stacked.mean(axis=0).astype(np.uint8)

    # FASE 1: Crop Global
    grad_x = cv2.Sobel(mean_frame, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mean_frame, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    _, edge_mask = cv2.threshold(magnitude.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)

    variance_map = stacked.std(axis=0)
    _, motion_mask = cv2.threshold(variance_map.astype(np.uint8), 3, 255, cv2.THRESH_BINARY)

    combined_mask = cv2.bitwise_or(edge_mask, motion_mask)
    kernel = np.ones((9, 9), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h_f, w_f = mean_frame.shape
    valid_contours = [c for c in contours if cv2.boundingRect(c)[2] > 100 and cv2.boundingRect(c)[3] > 100]

    if not valid_contours:
        return None

    x_min, y_min = w_f, h_f
    x_max, y_max = 0, 0

    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    w_phase1 = x_max - x_min
    h_phase1 = y_max - y_min

    # FASE 2: Micro-Crop de Precisió
    cropped_subframe = mean_frame[y_min:y_max, x_min:x_max]
    top_trim, bottom_trim, left_trim, right_trim = _refine_micro_borders(cropped_subframe)

    final_x = x_min + left_trim
    final_y = y_min + top_trim
    final_w = w_phase1 - left_trim - right_trim
    final_h = h_phase1 - top_trim - bottom_trim

    if top_trim or bottom_trim or left_trim or right_trim:
        print(f"🔍 Micro-crop Fase 2 aplicat: Top={top_trim}px, Bottom={bottom_trim}px, Left={left_trim}px, Right={right_trim}px", flush=True)

    return (final_x, final_y, final_w, final_h)


def generate_header_card_image(headline_html, width=960):
    """
    Renderitzador de capçaleres amb parsejador de negretes HTML (<b>...</b>) millorat
    i salt de línia dinàmic amb la font Lexend.
    """
    setup_fonts()

    font_name = ImageFont.truetype(LEXEND_BOLD_PATH, 36)
    font_handle = ImageFont.truetype(LEXEND_REGULAR_PATH, 28)
    
    # Mida de font principal per al títol
    font_size = 38
    font_text_reg = ImageFont.truetype(LEXEND_REGULAR_PATH, font_size)
    font_text_bold = ImageFont.truetype(LEXEND_BOLD_PATH, font_size)

    logo_size = 90
    if os.path.exists(LOGO_PATH):
        logo_img = Image.open(LOGO_PATH).convert("RGBA")
        logo_img = logo_img.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
    else:
        logo_img = Image.new("RGBA", (logo_size, logo_size), (200, 200, 200, 255))

    # Parsejador d'etiquetes <b> i </b>
    tokens = []
    # Divideix mantenint la distinció de si està dins d'un <b> o no
    raw_segments = re.split(r'(<b>.*?</b>)', headline_html, flags=re.IGNORECASE)
    for segment in raw_segments:
        if not segment:
            continue
        if segment.lower().startswith("<b>") and segment.lower().endswith("</b>"):
            clean_text = segment[3:-4]
            for w in clean_text.split():
                tokens.append((w, True))
        else:
            for w in segment.split():
                tokens.append((w, False))

    # Calculador de salts de línia per evitar que el text surti de la imatge
    lines = []
    current_line = []
    current_w = 0
    space_w = font_text_reg.getlength(" ")

    for word, is_bold in tokens:
        f = font_text_bold if is_bold else font_text_reg
        w_len = f.getlength(word)
        if current_w + w_len > (width - 40):
            if current_line:
                lines.append(current_line)
            current_line = [(word, is_bold)]
            current_w = w_len + space_w
        else:
            current_line.append((word, is_bold))
            current_w += w_len + space_w
            
    if current_line:
        lines.append(current_line)

    line_height = 50
    text_section_h = len(lines) * line_height
    header_h = 110
    total_h = header_h + text_section_h + 20

    final_card = Image.new("RGBA", (width, total_h), (0, 0, 0, 0))
    final_card.paste(logo_img, (10, 10), logo_img)

    draw = ImageDraw.Draw(final_card)
    draw.text((120, 15), "Feedity", font=font_name, fill=(255, 255, 255, 255))
    draw.text((120, 60), "@feedity.tv", font=font_handle, fill=(160, 160, 160, 255))

    y_cursor = header_h
    for line in lines:
        x_cursor = 10
        for word, is_bold in line:
            f = font_text_bold if is_bold else font_text_reg
            draw.text((x_cursor, y_cursor), word, font=f, fill=(255, 255, 255, 255))
            x_cursor += f.getlength(word) + space_w
        y_cursor += line_height

    img_path = "/tmp/header_card.png"
    final_card.save(img_path)
    return img_path, total_h


def process_video_canvas(input_path, headline, output_path="final_feedity.mp4"):
    print("🎬 Renderitzant composició del vídeo sobre el fons verd...", flush=True)
    clip = VideoFileClip(input_path)
    bbox = crop_content_bounding_box(clip)

    if bbox:
        x, y, w, h = bbox
        print(f"✂️ Marc extern eliminat: x={x}, y={y}, w={w}, h={h}", flush=True)
        cropped_clip = clip.cropped(x1=x, y1=y, x2=x + w, y2=y + h)
    else:
        cropped_clip = clip

    target_width = 960
    scaled_clip = cropped_clip.resized(width=target_width)

    bg_clip = ColorClip(size=(CANVAS_WIDTH, CANVAS_HEIGHT), color=CANVAS_BG_COLOR, duration=clip.duration)
    layers = [bg_clip]

    final_headline = headline if (headline and headline.strip()) else "<b>Featured</b> moment"

    header_img_path, card_h = generate_header_card_image(final_headline, width=target_width)
    header_clip = (
        ImageClip(header_img_path)
        .with_duration(clip.duration)
        .with_position(("center", 140))
    )
    layers.append(header_clip)

    video_y_pos = 140 + card_h + 20
    scaled_clip = scaled_clip.with_position(("center", video_y_pos))
    layers.append(scaled_clip)

    final_clip = CompositeVideoClip(layers, size=(CANVAS_WIDTH, CANVAS_HEIGHT))
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    clip.close()
    cropped_clip.close()
    final_clip.close()


def send_telegram_notification(video_path, headline, credits, generated_caption, video_id):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram configuració no trobada. Notificació omessa.", flush=True)
        return
        
    print("📤 Enviant notificació i vídeo a Telegram...", flush=True)
    message_text = (
        f"🎬 **NOU VÍDEO PROCESSAT PER A FEEDITY**\n\n"
        f"📌 **Títol del vídeo**: {headline}\n"
        f"👤 **Crèdits originals**: {credits}\n"
        f"🆔 **ID**: {video_id}\n\n"
        f"📝 **CAPTION GENERAT PER A PUBLICAR**:\n"
        f"```text\n{generated_caption}\n```"
    )
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    with open(video_path, "rb") as video_file:
        files = {"video": video_file}
        data = {
            "chat_id": TELEGRAM_CHAT_ID, 
            "caption": message_text,
            "parse_mode": "Markdown"
        }
        requests.post(url, data=data, files=files)


def main():
    print("🚀 Iniciant pipeline de Feedity...", flush=True)

    if not os.path.exists("sources.csv"):
        print("❌ ERROR: No s'ha trobat el fitxer sources.csv", flush=True)
        return

    processed = load_processed_ids()
    print(f"📊 Històric de vídeos processats anteriorment: {len(processed)}", flush=True)

    with open("sources.csv", "r") as f:
        reel_urls = [
            line.strip() for line in f
            if line.strip() and not line.startswith("reel_url") and not line.startswith("account_handle")
        ]

    if not reel_urls:
        print("⚠️ El fitxer sources.csv no conté cap URL per processar.", flush=True)
        return

    print(f"🔗 S'han trobat {len(reel_urls)} URLs a sources.csv", flush=True)

    processed_any = False
    for reel_url in reel_urls:
        print(f"\n🚀 Processant reel: {reel_url}", flush=True)
        video_file, video_id, credits, headline, generated_caption = get_reel_by_url(reel_url)

        if video_file:
            process_video_canvas(video_file, headline, "final_feedity.mp4")
            send_telegram_notification("final_feedity.mp4", headline, credits, generated_caption, video_id)
            save_processed_id(video_id)
            print("✅ Procés finalitzat amb èxit!", flush=True)
            processed_any = True
            break

    if not processed_any:
        print("ℹ️ No s'ha processat cap vídeo nou (tots ja estaven processats o s'ha produït un error).", flush=True)


if __name__ == "__main__":
    main()
