import os
import re
import json
import glob
import requests
import numpy as np
import cv2
import ollama
import yt_dlp
from moviepy import VideoFileClip, CompositeVideoClip

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# Opcional: ruta a un cookies.txt exportat del navegador, per si Instagram
# comença a exigir sessió iniciada per servir el vídeo (login required / rate limit).
INSTAGRAM_COOKIES_FILE = os.getenv("INSTAGRAM_COOKIES_FILE")
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
            model='gemma2',
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
    """Processa un reel a partir de la seva URL directa (mode semiautomàtic), via yt-dlp."""
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
    except yt_dlp.utils.DownloadError as e:
        print(f"❌ Error en descarregar el reel amb yt-dlp: {e}")
        print("   💡 Si l'error menciona 'login required' o 'rate-limit', prova a definir "
              "INSTAGRAM_COOKIES_FILE amb un cookies.txt exportat del navegador.")
        return None, None, None, None, None
    except Exception as e:
        print(f"❌ Error inesperat amb yt-dlp: {e}")
        return None, None, None, None, None

    video_id = str(info.get("id") or shortcode or reel_url)
    if video_id in processed_ids:
        print(f"⏭️ Reel ja processat anteriorment ({video_id}), s'omet.")
        return None, None, None, None, None

    downloaded_path = ydl.prepare_filename(info)
    # merge_output_format pot canviar l'extensió final a .mp4
    if not os.path.exists(downloaded_path):
        candidates = glob.glob("temp_input.*")
        downloaded_path = candidates[0] if candidates else None

    if not downloaded_path or not os.path.exists(downloaded_path):
        print("⚠️ yt-dlp no ha generat cap fitxer de vídeo.")
        return None, None, None, None, None

    caption_raw = info.get("description") or ""

    print("🤖 Analitzant i generant contingut amb Gemma 2...")
    credits, headline, generated_caption = analyze_caption_with_local_ai(caption_raw)

    return downloaded_path, video_id, credits, headline, generated_caption

    
def crop_content_bounding_box(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return (x, y, w, h)
    return None

def process_video_canvas(input_path, output_path="final_feedity.mp4"):
    clip = VideoFileClip(input_path)
    bbox = crop_content_bounding_box(input_path)
    
    if bbox and bbox[2] > 100 and bbox[3] > 100:
        x, y, w, h = bbox
        print(f"✂️ Crop aplicat: x={x}, y={y}, w={w}, h={h}")
        cropped_clip = clip.cropped(x1=x, y1=y, x2=x+w, y2=y+h)
    else:
        cropped_clip = clip

    scaled_clip = cropped_clip.resized(width=1080)
    
    final_clip = CompositeVideoClip(
        [scaled_clip.with_position("center")],
        size=(1080, 1920)
    )
    
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()
    cropped_clip.close()
    final_clip.close()

def send_telegram_notification(video_path, headline, credits, generated_caption, video_id):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Notificació de Telegram omessa (tokens no configurats).")
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
            process_video_canvas(video_file, "final_feedity.mp4")
            send_telegram_notification("final_feedity.mp4", headline, credits, generated_caption, video_id)
            save_processed_id(video_id)
            print("✅ Procés finalitzat amb èxit!")
            break
        else:
            print(f"⚠️ Reel omès o sense contingut nou: {reel_url}")

if __name__ == "__main__":
    main()
