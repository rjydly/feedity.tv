import os
import csv
import json
import subprocess

TODAY_QUEUE_FILE = 'today_queue.json'
BACKUP_CSV = 'backup_reels.csv'
SOURCES_CSV = 'sources.csv'


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


def get_next_reel_url():
    """Tria la URL del proper Reel a publicar (de la cua d'avui o del backup)."""
    # 1. Intentar agafar de la Cua d'Avui
    queue = load_today_queue()
    if queue:
        item = queue.pop(0)
        save_today_queue(queue)
        print(f"🎯 [CUA D'AVUI] Seleccionat Reel: {item['url']} ({item.get('likes', 0)} likes)")
        return item['url']

    # 2. Si la cua d'avui està buida, agafar el millor pendent de backup_reels.csv
    print("🛡️ [BACKUP MODE] La cua d'avui està buida. Cercant el Reel més viral pendent a backup_reels.csv...")
    rows = load_backup_csv()
    for r in rows:
        if r.get('status', '') == '':
            print(f"🎯 [BACKUP] Seleccionat Reel: {r['link']} ({r.get('likes', 0)} likes)")
            return r['link']

    print("⚠️ No hi ha cap Reel disponible ni a today_queue.json ni a backup_reels.csv.")
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

    # Executa el teu script main.py (amb Gemini/Groq, Plus Jakarta Sans, Portada i Buffer)
    print("🎬 Executant main.py...")
    subprocess.run(["python", "main.py"], check=True)


if __name__ == "__main__":
    main()