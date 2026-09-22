import os
import re
import csv
import json
import subprocess

TODAY_QUEUE_FILE = 'today_queue.json'
BACKUP_CSV = 'backup_reels.csv'
SOURCES_CSV = 'sources.csv'
DB_FILE = 'processed_videos.json'


def extract_shortcode(reel_url):
    match = re.search(r"instagram\.com/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)", reel_url)
    return match.group(1) if match else reel_url.strip()


def load_processed_ids():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def load_today_queue():
    if os.path.exists(TODAY_QUEUE_FILE):
        try:
            with open(TODAY_QUEUE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_today_queue(data):
    with open(TODAY_QUEUE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def load_backup_csv():
    if not os.path.exists(BACKUP_CSV):
        return []
    rows = []
    with open(BACKUP_CSV, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_backup_csv(rows):
    fieldnames = ['link', 'likes', 'status']
    with open(BACKUP_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def get_next_reel_url():
    """Tria la URL del proper Reel a publicar (cua d'avui o backup no processat)."""
    processed_ids = load_processed_ids()

    # 1. Intentar agafar de la Cua d'Avui
    queue = load_today_queue()
    while queue:
        item = queue.pop(0)
        save_today_queue(queue)
        item_id = str(item.get('id', ''))
        shortcode = extract_shortcode(item.get('url', ''))

        if item_id in processed_ids or shortcode in processed_ids:
            print(f"⏩ Ometent {item['url']} de today_queue (ja processat anteriorment).")
            continue

        print(f"🎯 [CUA D'AVUI] Seleccionat Reel: {item['url']} ({item.get('likes', 0)} likes)")
        return item['url']

    # 2. Si la cua d'avui està buida, cercar el més viral a backup_reels.csv
    print("🛡️ [BACKUP MODE] La cua d'avui està buida. Cercant a backup_reels.csv...")
    rows = load_backup_csv()
    updated = False
    chosen_url = None

    for r in rows:
        url = r.get('link', '').strip()
        if not url:
            continue
        shortcode = extract_shortcode(url)

        # Si ja s'ha processat en el passat, actualitzem el seu estat per netejar el CSV
        if shortcode in processed_ids:
            if r.get('status') != 'done':
                r['status'] = 'done'
                updated = True
            continue

        # Si està pendent (sense status) i no s'ha processat mai
        if not r.get('status', '').strip() and not chosen_url:
            chosen_url = url
            r['status'] = 'done'  # El marquem per no repetir-lo mai
            updated = True
            print(f"🎯 [BACKUP] Seleccionat Reel: {url} ({r.get('likes', 0)} likes)")
            break

    if updated:
        save_backup_csv(rows)

    if chosen_url:
        return chosen_url

    print("⚠️ No hi ha cap Reel disponible pendent ni a today_queue.json ni a backup_reels.csv.")
    return None


def set_sources_csv(reel_url):
    """Posa la URL a sources.csv com a 'pending' per a que main.py la processi."""
    with open(SOURCES_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['reel_url', 'status'])
        writer.writerow([reel_url, 'pending'])
    print(f"📝 sources.csv actualitzat amb: {reel_url} -> pending")


def main():
    reel_url = get_next_reel_url()
    if not reel_url:
        print("❌ No hi ha cap vídeo per publicar en aquesta execució.")
        return

    set_sources_csv(reel_url)

    print("🎬 Executant main.py...")
    subprocess.run(["python", "main.py"], check=True)


if __name__ == "__main__":
    main()
