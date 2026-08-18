import os
import re
import json
import glob
import requests
import numpy as np
import cv2
import ollama
import yt_dlp
from moviepy import VideoFileClip, CompositeVideoClip, TextClip

# Font utilitzada per "cremar" el titular sobre el vídeo.
HEADLINE_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
INSTAGRAM_COOKIES_FILE = os.getenv("INSTAGRAM_COOKIES_FILE")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
TEST_MODE = False


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
    1. Els crèdits o l'autor original del vídeo (p. ex., @usuari). Si no n'hi ha, posa "Unknown".
    2. Un títol o headline impactant de màxim 6 paraules en anglès per posar a sobre del vídeo.
    3. Una nova descripció (caption) optimitzada per a Instagram/TikTok en anglès, amb crida a l'acció (CTA) i hashtags virals.

    Descripció original: "{caption}"

    Respon NOMÉS en format JSON com aquest:
    {{"credits": "@usuari", "headline": "Titular Impactant", "generated_caption": "Text de la descripció nova amb hashtags..."}}
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        content = response['message']['content']
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return (
                data.get("credits", "Unknown"), 
                data.get("headline", ""), 
                data.get("generated_caption", "")
            )
    except Exception as e:
        print(f"⚠️ Error en analitzar amb Ollama: {e}")
    
    return "Unknown", "", caption


def extract_shortcode(reel_url):
    """Extreu el shortcode (p.ex. 'CxYz123AbCd') d'una URL de reel/post d'Instagram."""
    match = re.search(r"instagram\.com/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)", reel_url)
    return match.group(1) if match else None


def _cleanup_temp_input():
    """Elimina restes de descàrregues anteriors (temp_input.* de qualsevol extensió)."""
    for f in glob.glob("temp_input.*"):
        try:
            os.remove(f)
        except OSError:
            pass


def get_reel_by_url(reel_url):
    """Processa un reel a partir de la seva URL directa, via yt-dlp."""
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
        print("   💡 Si l'error menciona 'login required' o 'rate-limit', assegura't que "
              "INSTAGRAM_COOKIES_FILE conté un cookies.txt vàlid.")
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

    print(f"🤖 Analitzant i generant contingut amb Ollama ({OLLAMA_MODEL})...")
    credits, headline, generated_caption = analyze_caption_with_local_ai(caption_raw)

    return downloaded_path, video_id, credits, headline, generated_caption


# ==========================================
# DETECCIÓ I CROP INTEL·LIGENT DE CONTINGUT
# ==========================================

def detect_background_color(frame):
    """Detecta el color de fons mostrejant els extrems superior i inferior centrals."""
    h, w, _ = frame.shape
    top_sample = frame[0:15, w//4:3*w//4]
    bottom_sample = frame[h-15:h, w//4:3*w//4]
    samples = np.concatenate([top_sample, bottom_sample], axis=0)
    return np.median(samples, axis=(0, 1))


def get_longest_consecutive_run(bool_array):
    """Calcula la seqüència contínua més llarga de valors True en un array 1D."""
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
    """
    Troba el requadre rectangular del vídeo real ignorant text, avatars i fons plans.
    """
    h, w, _ = frame.shape
    bg_color = detect_background_color(frame)

    diff = np.max(np.abs(frame.astype(np.float32) - bg_color), axis=2)
    foreground_mask = diff > color_diff_threshold

    # 1. Escaneig Vertical (ignora text/logos que no ocupen prou amplada contínua)
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

    # 2. Escaneig Horitzontal dins de la franja y1:y2
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
    """
    Mostreja diversos fotogrames al llarg del vídeo i obté la mediana del requadre
    on hi ha contingut real en moviment/vídeo.
    """
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
# RENDERITZAT I EDICIÓ DE VÍDEO
# ==========================================

def build_headline_clip(headline, duration, canvas_width=1080):
    """Crea el TextClip del titular que es posiciona a sobre del vídeo."""
    if not headline:
        return None
    try:
        txt_clip = TextClip(
            font=HEADLINE_FONT,
            text=headline.upper(),
            font_size=80,
            color="white",
            stroke_color="black",
            stroke_width=4,
            method="caption",
            size=(int(canvas_width * 0.9), 320),
            text_align="center",
            horizontal_align="center",
            interline=20,
        )
        return txt_clip.with_duration(duration).with_position(("center", 140))
    except Exception as e:
        print(f"⚠️ No s'ha pogut generar el titular sobre el vídeo: {e}")
        return None


def process_video_canvas(input_path, headline, output_path="final_feedity.mp4"):
    clip = VideoFileClip(input_path)
    frame_w, frame_h = clip.w, clip.h
    bbox = crop_content_bounding_box(clip)

    min_area_ratio = 0.10
    if bbox and (bbox[2] * bbox[3]) >= min_area_ratio * frame_w * frame_h:
        x, y, w, h = bbox
        # Marge de seguretat del 1%
        margin_x = int(w * 0.01)
        margin_y = int(h * 0.01)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame_w, x + w + margin_x)
        y2 = min(frame_h, y + h + margin_y)
        print(f"✂️ Crop aplicat amb èxit: x={x1}, y={y1}, x2={x2}, y2={y2}")
        cropped_clip = clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
    else:
        print("ℹ️ No s'ha detectat cap marc distintiu, no s'aplica crop.")
        cropped_clip = clip

    scaled_clip = cropped_clip.resized(width=1080)
    layers = [scaled_clip.with_position("center")]

    headline_clip = build_headline_clip(headline, scaled_clip.duration)
    if headline_clip:
        print(f"🖋️ Sobreposant titular: {headline}")
        layers.append(headline_clip)

    final_clip = CompositeVideoClip(layers, size=(1080, 1920))
    
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
    if headline_clip:
        headline_clip.close()


# ==========================================
# NOTIFICACIÓ TELEGRAM
# ==========================================

def send_telegram_notification(video_path, headline, credits, generated_caption, video_id):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Notificació de Telegram omessa (tokens no configurats).")
        return
        
    message_text = (
        f"🎬 *NOU VÍDEO PROCESSAT PER A FEEDITY*\n\n"
        f"📌 *Títol*: {headline}\n"
        f"👤 *Crèdits*: {credits}\n"
        f"🆔 *ID*: `{video_id}`\n\n"
        f"📝 *CAPTION GENERAT*:\n{generated_caption}"
    )

    # Telegram sendVideo té un límit estricte de 1024 caràcters per al caption
    if len(message_text) > 1024:
        message_text = message_text[:1020] + "..."
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    with open(video_path, "rb") as video_file:
        files = {"video": video_file}
        data = {
            "chat_id": TELEGRAM_CHAT_ID, 
            "caption": message_text,
            "parse_mode": "Markdown"
        }
        res = requests.post(url, data=data, files=files)
        if res.status_code == 200:
            print("🚀 Vídeo i dades enviats correctament a Telegram!")
        else:
            print(f"❌ Error en enviar a Telegram: {res.text}")


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
        else:
            print(f"⚠️ Reel omès o sense contingut nou: {reel_url}")


if __name__ == "__main__":
    main()
