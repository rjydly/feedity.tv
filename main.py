import os
import re
import json
import requests
import numpy as np
import cv2
import ollama
from apify_client import ApifyClient
from moviepy import VideoFileClip, CompositeVideoClip

# Configurar variables des de les variables d'entorn
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TEST_MODE = False  # Canviar a True si només vols fer proves sense desar l'historial

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
    Extreu dues coses d'aquesta descripció de xarxes socials:
    1. Els crèdits o l'autor original del vídeo (p. ex., @usuari). Si no n'hi ha, posa "Unknown".
    2. Un titular o resum impactant de màxim 6 paraules en anglès.

    Descripció: "{caption}"

    Respon NOMÉS en format JSON com aquest:
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

def get_most_viral_video_from_account(account_handle):
    processed_ids = load_processed_ids()
    clean_handle = account_handle.replace("@", "").strip()
    
    print(f"📡 Scraping de @{clean_handle} via Apify...")
    
    client = ApifyClient(APIFY_TOKEN)
    
    run_input = {
        "directUrls": [f"https://www.instagram.com/{clean_handle}/"],
        "resultsType": "posts",
        "resultsLimit": 12,
        "searchType": "hashtag"
    }
    
    try:
        run = client.actor("apify/instagram-scraper").call(run_input=run_input)
        dataset_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    except Exception as e:
        print(f"❌ Error en executar l'Actor d'Apify: {e}")
        return None, None, None, None

    candidates = []
    for item in dataset_items:
        if item.get("type") != "Video" and not item.get("isVideo", False):
            continue
            
        video_id = str(item.get("id") or item.get("shortCode"))
        if video_id in processed_ids:
            continue
            
        views = item.get("videoViewCount") or item.get("likesCount") or 0
        video_url = item.get("videoUrl")
        caption_text = item.get("caption") or ""
        
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

def crop_content_bounding_box(video_path):
    """Detecta la regió del vídeo real ignorant els marcs negres i textos adjunts."""
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
        print(f"✂️ Crop Bounding Box aplicat: x={x}, y={y}, w={w}, h={h}")
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

def send_telegram_notification(video_path, caption):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Notificació de Telegram omessa (tokens no configurats).")
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
