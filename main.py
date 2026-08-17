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
# Canvia aquest valor quan vulguis modificar el color de fons del lienzo (RGB):
# Verd debug: (0, 255, 0) | Negre: (0, 0, 0) | Blanc: (255, 255, 255)
CANVAS_BG_COLOR = (0, 255, 0)

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
LOGO_PATH = "assets/logo.png"

# Descàrrega temporal de la font Lexend des de Google Fonts (no es descarrega al repo)
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
                print(f"⚠️ Error descarregant la font ({url}): {e}")

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
    prompt = f"""
    Analitza aquesta descripció de xarxes socials i genera:
    1. Els crèdits o l'autor original del vídeo (p. ex., @usuari). Si no s'identifica o no existeix, posa "Unknown".
    2. Un títol o headline en anglès o castellà. Utilitza etiquetes <b> i </b> per marcar les paraules o frases clau que s'han de posar en NEGRETA, tal com les etiquetes de text d'Instagram o Twitter.
    3. Una descripció (caption) llarga, detallada i ben estructurada per a la publicació d'Instagram, amb ganxo inicial, explicació del tema, crida a l'acció (CTA) i hashtags virals. NO incloguis els crèdits dins d'aquest camp de caption.

    Descripció original: "{caption}"

    Respon NOMÉS en format JSON vàlid com aquest:
    {{"credits": "@usuari", "headline": "Això és un <b>text impactant</b> en negreta", "generated_caption": "Texto largo y detallado para Instagram..."}}
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
            headline = data.get("headline", "")
            gen_caption = data.get("generated_caption", "")
            
            # Si s'ha trobat l'autor original, s'afegeix la font al final de tot de la descripció
            if credits and credits != "Unknown":
                gen_caption = f"{gen_caption}\n\nVia: {credits}"
                
            return credits, headline, gen_caption
    except Exception as e:
        print(f"⚠️ Error en analitzar amb Ollama: {e}")
    
    return "Unknown", "", caption

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
    if INSTAGRAM_COOKIES_FILE:
        ydl_opts["cookiefile"] = INSTAGRAM_COOKIES_FILE

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(reel_url, download=True)
    except Exception as e:
        print(f"❌ Error descarregant amb yt-dlp: {e}")
        return None, None, None, None, None

    video_id = str(info.get("id") or shortcode or reel_url)
    if video_id in processed_ids:
        print(f"⏭️ Reel ja processat ({video_id}), s'omet.")
        return None, None, None, None, None

    downloaded_path = ydl.prepare_filename(info)
    if not os.path.exists(downloaded_path):
        candidates = glob.glob("temp_input.*")
        downloaded_path = candidates[0] if candidates else None

    if not downloaded_path or not os.path.exists(downloaded_path):
        return None, None, None, None, None

    caption_raw = info.get("description") or ""

    print("🤖 Analitzant contingut amb Gemma 2...")
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

def crop_content_bounding_box(clip, std_threshold=4.0):
    """Detecta el contingut i elimina tant marcs negres estàtics com àrees sense moviment."""
    frames = _sample_frames_grayscale_from_clip(clip)
    if len(frames) < 2:
        return None

    # 1. Filtre per eliminar negre pur/fosc de les vores
    stacked = np.stack(frames, axis=0)
    mean_frame = stacked.mean(axis=0)
    _, black_mask = cv2.threshold(mean_frame.astype(np.uint8), 15, 255, cv2.THRESH_BINARY)

    # 2. Màscara de moviment
    variance_map = stacked.std(axis=0)
    motion_mask = (variance_map > std_threshold).astype(np.uint8) * 255

    # Combinem les dues màscares
    combined_mask = cv2.bitwise_and(black_mask, motion_mask)

    kernel = np.ones((15, 15), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Si no detecta moviment, almenys retallem el marc negre
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h)


def generate_header_card_image(headline_html, width=960):
    """Crea una imatge amb el logo, Feedity, @feedity.tv i el text amb la font Lexend i etiquetes <b>."""
    setup_fonts()

    # Mida de les fonts
    font_name = ImageFont.truetype(LEXEND_BOLD_PATH, 36)
    font_handle = ImageFont.truetype(LEXEND_REGULAR_PATH, 28)
    font_text_reg = ImageFont.truetype(LEXEND_REGULAR_PATH, 42)
    font_text_bold = ImageFont.truetype(LEXEND_BOLD_PATH, 42)

    # Imatge temporal en blanc per a mesurar l'alçada del text
    temp_img = Image.new("RGBA", (width, 2000), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_img)

    # Carrega el logo
    logo_size = 100
    if os.path.exists(LOGO_PATH):
        logo_img = Image.open(LOGO_PATH).convert("RGBA")
        logo_img = logo_img.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
    else:
        logo_img = Image.new("RGBA", (logo_size, logo_size), (200, 200, 200, 255))

    # Parseja les paraules i estil (normal / negreta)
    tokens = []
    parts = re.split(r'(<b>.*?</b>)', headline_html)
    for part in parts:
        if part.startswith("<b>") and part.endswith("</b>"):
            clean_text = part[3:-4]
            words = clean_text.split(" ")
            for w in words:
                if w: tokens.append((w, True))
        else:
            words = part.split(" ")
            for w in words:
                if w: tokens.append((w, False))

    # Calcula el salt de línia del text
    lines = []
    current_line = []
    current_w = 0
    space_w = font_text_reg.getlength(" ")

    for word, is_bold in tokens:
        f = font_text_bold if is_bold else font_text_reg
        w_len = f.getlength(word)
        if current_w + w_len > width - 40:
            lines.append(current_line)
            current_line = [(word, is_bold)]
            current_w = w_len + space_w
        else:
            current_line.append((word, is_bold))
            current_w += w_len + space_w
    if current_line:
        lines.append(current_line)

    line_height = 54
    text_section_h = len(lines) * line_height
    header_h = 130
    total_h = header_h + text_section_h + 30

    # Imatge final transparent
    final_card = Image.new("RGBA", (width, total_h), (0, 0, 0, 0))
    final_card.paste(logo_img, (10, 10), logo_img)

    draw = ImageDraw.Draw(final_card)
    draw.text((130, 20), "Feedity", font=font_name, fill=(255, 255, 255, 255))
    draw.text((130, 68), "@feedity.tv", font=font_handle, fill=(160, 160, 160, 255))

    # Renderitza les línies de text amb la font Lexend
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
    clip = VideoFileClip(input_path)
    frame_w, frame_h = clip.size
    bbox = crop_content_bounding_box(clip)

    min_area_ratio = 0.10
    if bbox and (bbox[2] * bbox[3]) >= min_area_ratio * frame_w * frame_h:
        x, y, w, h = bbox
        margin_x = int(w * 0.02)
        margin_y = int(h * 0.02)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame_w, x + w + margin_x)
        y2 = min(frame_h, y + h + margin_y)
        cropped_clip = clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
    else:
        cropped_clip = clip

    scaled_clip = cropped_clip.resized(width=1000)

    # Crea el fons (fàcil de canviar amb la variable CANVAS_BG_COLOR)
    bg_clip = ColorClip(size=(CANVAS_WIDTH, CANVAS_HEIGHT), color=CANVAS_BG_COLOR, duration=clip.duration)

    layers = [bg_clip]

    if headline:
        header_img_path, card_h = generate_header_card_image(headline, width=1000)
        header_clip = (
            ImageClip(header_img_path)
            .with_duration(clip.duration)
            .with_position(("center", 180))
        )
        layers.append(header_clip)
        video_y_pos = 180 + card_h + 30
    else:
        video_y_pos = 300

    scaled_clip = scaled_clip.with_position(("center", video_y_pos))
    layers.append(scaled_clip)

    final_clip = CompositeVideoClip(layers, size=(CANVAS_WIDTH, CANVAS_HEIGHT))
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    clip.close()
    cropped_clip.close()
    final_clip.close()

def send_telegram_notification(video_path, headline, credits, generated_caption, video_id):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Notificació de Telegram omessa.")
        return
        
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
        video_file, video_id, credits, headline, generated_caption = get_reel_by_url(reel_url)

        if video_file:
            process_video_canvas(video_file, headline, "final_feedity.mp4")
            send_telegram_notification("final_feedity.mp4", headline, credits, generated_caption, video_id)
            save_processed_id(video_id)
            print("✅ Procés finalitzat amb èxit!")
            break

if __name__ == "__main__":
    main()
