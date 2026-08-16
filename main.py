import os
import re
import json
import requests
import numpy as np
import cv2
import ollama
from moviepy import VideoFileClip, CompositeVideoClip, ImageClip

# Configurar variables des del workflow
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TEST_MODE = True  # Canviar a True si només vols fer proves sense desar historial

# Carregar historial de vídeos ja processats
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

# Analitzar el text del Reel amb la IA local (Gemma 2)
def analyze_caption_with_local_ai(caption):
    prompt = f"""
    Extreu dues coses d'aquesta descripció de xarxes socials:
    1. Els crèdits o l'autor original del vídeo (p. ex., @usuari). Si no n'hi ha, posa "Unknown".
    2. Un titular o resum impactant de màxim 6 paraules en anglès.

    Descripció: "{caption}"

    Respon NORMÉS en format JSON com aquest:
    {{"credits": "@usuari", "headline": "Titular Impactant Del Video"}}
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
            return data.get("credits", "Unknown"), data.get("headline", "")
    except Exception as e:
        print(f"⚠️ Error en analitzar amb Ollama: {e}")
    
    return "Unknown", ""

# Extreure el Reel més viral des de RapidAPI
def get_most_viral_video_from_account(account_handle):
    processed_ids = load_processed_ids()
    clean_handle = account_handle.replace("@", "").strip()
    
    print(f"📡 Obtenint Reels de @{clean_handle} via RapidAPI...")
    
    url = "https://website-social-scraper-api.p.rapidapi.com/get-social-details"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "website-social-scraper-api.p.rapidapi.com"
    }
    params = {"username": clean_handle}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            print(f"❌ Error API ({response.status_code}): {response.text}")
            return None, None, None, None
            
        data = response.json()
        items = data.get("posts", []) or data.get("data", [])
    except Exception as e:
        print(f"❌ Error de connexió amb RapidAPI: {e}")
        return None, None, None, None

    candidates = []
    for node in items:
        if not node.get("is_video", True):
            continue
            
        video_id = str(node.get("id") or node.get("shortcode"))
        if video_id in processed_ids:
            continue
            
        views = node.get("video_view_count") or node.get("like_count", 0)
        video_url = node.get("video_url")
        caption_text = node.get("caption") or ""
        
        if video_url:
            candidates.append({
                "id": video_id,
                "url": video_url,
                "views": views,
                "caption": caption_text
            })
        
    candidates.sort(key=lambda x: x["views"], reverse=True)
    
    for item in candidates:
        video_id = item["id"]
        video_url = item["url"]
        caption_raw = item["caption"]
        
        print(f"🔥 Vídeo candidat trobat ({video_id}) amb {item['views']} reproduccions.")
        
        r = requests.get(video_url, stream=True)
        with open("temp_input.mp4", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    
        print("🤖 Analitzant caption amb Gemma...")
        credits, headline = analyze_caption_with_local_ai(caption_raw)
        
        return "temp_input.mp4", video_id, credits, headline
        
    return None, None, None, None

# Processament del vídeo a format 9:16 vertical
def process_video_canvas(input_path, output_path="output.mp4"):
    clip = VideoFileClip(input_path)
    
    # Comprovar si el vídeo ja és vertical o si cal encadirar-lo
    w, h = clip.size
    if h > w:
        print("📐 El vídeo ja és vertical. Ajustant directament a 1080x1920...")
        final_clip = clip.resized(height=1920) if clip.h != 1920 else clip
    else:
        print("📐 Convertint vídeo horitzontal a canvas 9:16...")
        # Centrar el vídeo sobre un fons de 1080x1920
        scaled_clip = clip.resized(width=1080)
        final_clip = CompositeVideoClip([scaled_clip.with_position("center")], size=(1080, 1920))
    
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    clip.close()
    final_clip.close()

# Enviar el vídeo acabat per Telegram
def send_telegram_notification(video_path, caption):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Notificació de Telegram omessa (manten fitxers no configurats).")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    with open(video_path, "rb") as video_file:
        files = {"video": video_file}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        res = requests.post(url, data=data, files=files)
        if res.status_code == 200:
            print("🚀 Vídeo enviat correctament a Telegram!")
        else:
            print(f"❌ Error en enviar a Telegram: {res.text}")

def main():
    if not os.path.exists("sources.csv"):
        print("❌ No s'ha trobat el fitxer sources.csv")
        return

    with open("sources.csv", "r") as f:
        accounts = [line.strip() for line in f if line.strip() and not line.startswith("account_handle")]

    for account in accounts:
        print(f"\n🚀 Processant compte: {account}")
        video_file, video_id, credits, headline = get_most_viral_video_from_account(account)
        
        if video_file:
            process_video_canvas(video_file, "final_feedity.mp4")
            
            telegram_text = f"✨ **Nou contingut a punt**\n\n📌 **Headline**: {headline}\n👤 **Crèdits**: {credits}\n🆔 **ID**: {video_id}"
            send_telegram_notification("final_feedity.mp4", telegram_text)
            
            save_processed_id(video_id)
            print("✅ Procés finalitzat amb èxit!")
            break
        else:
            print(f"⚠️ No s'han trobat nous vídeos no processats per a {account}.")

if __name__ == "__main__":
    main()
