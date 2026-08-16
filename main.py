import os
import json
import csv
import subprocess
import requests
import cv2
import numpy as np
import ollama
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, CompositeVideoClip, ImageClip


# ==========================================
# CONFIGURACIÓ I VARIABLES D'ENTORN
# ==========================================
TEST_MODE = True  # Canvia a False per a producció

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "EL_TEU_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "EL_TEU_CHAT_ID")

PROCESSED_FILE = "processed_videos.json"
ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
OUTPUT_VIDEO_PATH = "output_feedity.mp4"

# ==========================================
# 1. GESTIÓ DE VÍDEOS PROCESSATS
# ==========================================
def load_processed_ids():
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_processed_id(video_id):
    if TEST_MODE:
        print("🧪 [TEST_MODE] Mode de prova actiu: No s'enregistra l'ID al fitxer JSON.")
        return
    processed = load_processed_ids()
    processed.add(video_id)
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed), f, indent=4)

# ==========================================
# 2. ANÀLISI DE CAPTION AMB IA LOCAL (OLLAMA)
# ==========================================
def analyze_caption_with_local_ai(caption_text):
    """
    Crida a Ollama en local per analitzar el caption, extreure els crèdits originals
    i generar un titular breu.
    """
    if not caption_text:
        return "Via @wealth", "Viral Clip"

    prompt = f"""
    Analitza el següent text d'una publicació de xarxes socials.
    Anomena la font original o autor del contingut (busca mencions com 'cr:', 'via:', 'cc:' o usuaris amb '@').
    I crea un titular o resumeix la idea en 3 a 5 paraules.
    
    TEXT ORIGINAL:
    "{caption_text}"
    
    Respon ÚNICAMENT en un JSON vàlid amb aquesta estructura:
    {{
        "credits": "Via @nom_de_usuari",
        "headline": "Titular Breu I Impactant"
    }}
    """

    try:
        response = ollama.chat(
            model="gemma2",
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        data = json.loads(response['message']['content'])
        credits = data.get("credits", "Via @wealth")
        headline = data.get("headline", "")
        return credits, headline
    except Exception as e:
        print(f"⚠️ Error executant la IA local (Ollama): {e}")
        return "Via @wealth", ""

# ==========================================
# 3. CERCA DEL MÉS VIRAL I DESCARREGA
# ==========================================
def get_most_viral_video_from_account(account_handle):
    processed_ids = load_processed_ids()
    url = f"https://www.instagram.com/{account_handle.replace('@', '')}/"
    
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-single-json",
        "--playlist-end", "15",
        url
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ Error en llegir el compte {account_handle}")
        return None, None, None, None
        
    data = json.loads(res.stdout)
    entries = data.get("entries", [])
    
    valid_entries = [e for e in entries if e.get("id") not in processed_ids]
    sorted_entries = sorted(
        valid_entries, 
        key=lambda x: x.get("like_count") or x.get("view_count") or 0, 
        reverse=True
    )
    
    for entry in sorted_entries:
        video_id = entry.get("id")
        video_url = entry.get("url") or entry.get("webpage_url")
        print(f"🔍 Avaluant vídeo {video_id} de {account_handle}...")
        
        dl_cmd = [
            "yt-dlp",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--write-description",
            "-o", "temp_input.%(ext)s",
            video_url
        ]
        subprocess.run(dl_cmd, check=True)
        
        caption_raw = ""
        if os.path.exists("temp_input.description"):
            with open("temp_input.description", "r", encoding="utf-8") as f:
                caption_raw = f.read()
                
        print("🤖 Processant el caption amb la IA local (Gemma)...")
        credits, headline = analyze_caption_with_local_ai(caption_raw)
        
        return "temp_input.mp4", video_id, credits, headline
        
    return None, None, None, None

# ==========================================
# 4. VALIDAció I CÀLCUL DE BBOX (OPENCV)
# ==========================================
def detect_and_validate_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        return False, None
        
    h_img, w_img, _ = frame.shape
    total_area = h_img * w_img
    aspect_ratio = w_img / float(h_img)
    
    # REGLA 1: Descartar si el vídeo font ja és vertical complet (9:16)
    if 0.50 <= aspect_ratio <= 0.62:
        print("❌ Descartat: El vídeo ja té format vertical complet 9:16.")
        return False, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_bbox = None
    max_area = 0
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ar = w / float(h)
        
        # El marc del vídeo ha de tenir una mida raonable
        if area > (total_area * 0.20) and (0.8 <= ar <= 2.2):
            if area > max_area:
                max_area = area
                best_bbox = (x, y, w, h)
                
    if best_bbox is None:
        print("❌ Descartat: No s'ha pogut determinar el marc del vídeo interior.")
        return False, None
        
    bx, by, bw, bh = best_bbox
    if bw >= (w_img * 0.98) and bh >= (h_img * 0.98):
        print("❌ Descartat: Marca d'enquadrament dubtosa (ocupa el 100% del frame).")
        return False, None
        
    print(f"✅ Enquadrament detectat: x={bx}, y={by}, w={bw}, h={bh}")
    return True, best_bbox

# ==========================================
# 5. GENERACIÓ DEL CANVAS 9:16 (MOVIEPY / PIL)
# ==========================================
def render_feedity_canvas(video_path, bbox, credits_text, headline_text):
    x, y, w, h = bbox
    clip = VideoFileClip(video_path)
    
    # Retallat segons les coordenades exactes en píxels
    cropped_clip = clip.crop(x1=x, y1=y, width=w, height=h)
    
    canvas_w, canvas_h = 1080, 1920
    scaled_clip = cropped_clip.resize(width=canvas_w)
    
    video_y_position = (canvas_h - scaled_clip.h) // 2
    positioned_video = scaled_clip.set_position(("center", video_y_position))
    
    # Fons fosc Pubity-style
    overlay_img = Image.new("RGBA", (canvas_w, canvas_h), (15, 15, 15, 255))
    draw = ImageDraw.Draw(overlay_img)
    
    # Carregar logo transparent des de assets/
    if os.path.exists(LOGO_PATH):
        logo = Image.open(LOGO_PATH).convert("RGBA")
        logo = logo.resize((120, 120))
        overlay_img.paste(logo, ((canvas_w - 120) // 2, 120), logo)
        
    # Fonts
    try:
        font_credits = ImageFont.truetype("Lexend-Regular.ttf", 28)
        font_headline = ImageFont.truetype("Lexend-Bold.ttf", 40)
    except:
        font_credits = ImageFont.load_default()
        font_headline = ImageFont.load_default()
        
    # Dibuixar Titular (A dalt del vídeo, sota el logo)
    if headline_text:
        draw.text((canvas_w // 2, 280), headline_text.upper(), fill=(255, 255, 255), font=font_headline, anchor="ms")
        
    # Dibuixar Font / Crèdits (A baix del vídeo)
    draw.text((canvas_w // 2, canvas_h - 150), f"Source: {credits_text}", fill=(180, 180, 180), font=font_credits, anchor="ms")
    
    overlay_img.save("temp_overlay.png")
    overlay_clip = ImageClip("temp_overlay.png").set_duration(clip.duration)
    
    final_clip = CompositeVideoClip([overlay_clip, positioned_video], size=(canvas_w, canvas_h))
    final_clip.write_videofile(OUTPUT_VIDEO_PATH, fps=30, codec="libx264", audio_codec="aac")
    
    clip.close()
    final_clip.close()

# ==========================================
# 6. ENVIAMENT A TELEGRAM (TEST_MODE)
# ==========================================
def send_telegram_video(video_path, caption):
    print("📤 Enviant vídeo de prova a Telegram...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    with open(video_path, "rb") as video_file:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        files = {"video": video_file}
        response = requests.post(url, data=payload, files=files)
        
    if response.status_code == 200:
        print("✅ Vídeo enviat correctament a Telegram!")
    else:
        print(f"❌ Error enviant a Telegram: {response.text}")

# ==========================================
# EXECUCIÓ PRINCIPAL
# ==========================================
def main():
    sources = []
    if os.path.exists("sources.csv"):
        with open("sources.csv", mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            sources = [row["account_handle"] for row in reader if row.get("account_handle")]
            
    if not sources:
        sources = ["@wealth"]
        
    for account in sources:
        print(f"\n🚀 Processant compte: {account}")
        video_file, video_id, credits, headline = get_most_viral_video_from_account(account)
        
        if not video_file:
            print("⚠️ No s'han trobat nous vídeos virals per analitzar.")
            continue
            
        is_valid, bbox = detect_and_validate_video(video_file)
        
        if is_valid:
            print("🎨 Generant canvas 9:16 i retallant el vídeo...")
            render_feedity_canvas(video_file, bbox, credits, headline)
            
            caption_text = f"📹 Feedity TV Preview\n\n📌 Account: {account}\n👤 Credits: {credits}\n💡 Headline: {headline}\n🆔 Video ID: {video_id}"
            
            if TEST_MODE:
                send_telegram_video(OUTPUT_VIDEO_PATH, caption_text)
            else:
                save_processed_id(video_id)
                print("🚀 [PROD] Vídeo preparat per a la publicació automàtica.")
            break

if __name__ == "__main__":
    main()
