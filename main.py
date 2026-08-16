import os
import re
import json
import requests
import numpy as np
import cv2
import ollama
from apify_client import ApifyClient
from moviepy import VideoFileClip, CompositeVideoClip

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
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

def get_most_viral_video_from_account(account_handle):
    processed_ids = load_processed_ids()
    clean_handle = account_handle.replace("@", "").strip()
    
    print(f"📡 Scraping de @{clean_handle} via Apify...")
    
    client = ApifyClient(APIFY_TOKEN)
    
    # Paràmetres oficials recomanats per a apify/instagram-scraper
    run_input = {
        "usernames": [clean_handle],
        "resultsType": "posts",
        "resultsLimit": 20,
        "addParentData": False
    }
    
    try:
        run = client.actor("apify/instagram-scraper").call(run_input=run_input)
        dataset_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    except Exception as e:
        print(f"❌ Error en executar l'Actor d'Apify: {e}")
        return None, None, None, None, None

    candidates = []
    for item in dataset_items:
        # Acceptem qualsevol ítem que porti URL de vídeo directa
        video_url = item.get("videoUrl")
        if not video_url:
            continue
            
        video_id = str(item.get("id") or item.get("shortCode"))
        if video_id in processed_ids:
            continue
            
        views = item.get("videoViewCount") or item.get("playCount") or item.get("likesCount") or 0
        caption_text = item.get("caption") or ""
        
        candidates.append({
            "id": video_id,
            "url": video_url,
            "views": views,
            "caption": caption_text
        })
        
    print(f"📊 Troba {len(candidates)} vídeos potencials no processats.")
    candidates.sort(key=lambda x: x["views"], reverse=True)
    
    for item in candidates:
        video_id = item["id"]
        video_url = item["url"]
        caption_raw = item["caption"]
        
        print(f"🔥 Vídeo candidat trobat ({video_id}) amb {item['views']} visualitzacions.")
        
        r = requests.get(video_url, stream=True)
        with open("temp_input.mp4", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    
        print("🤖 Analitzant i generant contingut amb Gemma 2...")
        credits, headline, generated_caption = analyze_caption_with_local_ai(caption_raw)
        
        return "temp_input.mp4", video_id, credits, headline, generated_caption
        
    return None, None, None, None, None

    
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
        accounts = [line.strip() for line in f if line.strip() and not line.startswith("account_handle")]

    for account in accounts:
        print(f"\n🚀 Processant compte: {account}")
        video_file, video_id, credits, headline, generated_caption = get_most_viral_video_from_account(account)
        
        if video_file:
            process_video_canvas(video_file, "final_feedity.mp4")
            send_telegram_notification("final_feedity.mp4", headline, credits, generated_caption, video_id)
            save_processed_id(video_id)
            print("✅ Procés finalitzat amb èxit!")
            break
        else:
            print(f"⚠️ No s'han trobat nous vídeos no processats per a {account}.")

if __name__ == "__main__":
    main()
