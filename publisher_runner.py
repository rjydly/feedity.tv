import os
import re
import csv
import json
import subprocess

TODAY_QUEUE_FILE = 'today_queue.json'
BACKUP_CSV = 'backup_reels.csv'
SOURCES_CSV = 'sources.csv'
DB_FILE = 'processed_videos.json'

MAX_RETRIES = 5  # Nombre màxim d'intents consecutius si un enllaç falla


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

    # 2. Si la cua d'avui està buida, cercar a backup_reels.csv
    print("🛡️ [BACKUP MODE] Cercant el següent Reel a backup_reels.csv...")
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

        # Si no té status (és a dir, no ha estat provat ni completat)
        if not r.get('status', '').strip() and not chosen_url:
            chosen_url = url
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


def get_sources_csv_status(reel_url):
    """Comprova si main.py ha marcat el vídeo com a 'done' o 'failed'."""
    if not os.path.exists(SOURCES_CSV):
        return ""
    try:
        with open(SOURCES_CSV, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and len(row) >= 2 and row[0].strip() == reel_url.strip():
                    return row[1].strip().lower()
    except Exception:
        return ""
    return ""


def main():
    print(f"🚀 Iniciant pipeline de publicació (fins a {MAX_RETRIES} intents si hi ha fallades)...")

    for attempt in range(1, MAX_RETRIES + 1):
        reel_url = get_next_reel_url()
        if not reel_url:
            print("❌ No queden més vídeos disponibles a la cua ni al backup.")
            return

        set_sources_csv(reel_url)

        print(f"\n🎬 [Intent {attempt}/{MAX_RETRIES}] Executant main.py per a: {reel_url}...")
        
        # Executem main.py sense check=True per capturar si falla i poder passar al següent
        subprocess.run(["python", "main.py"], check=False)

        # Comprovem el resultat escrit per main.py a sources.csv
        status = get_sources_csv_status(reel_url)
        if status == "done":
            print(f"\n🎉 Vídeo processat i publicat amb èxit a l'intent {attempt}!")
            return
        else:
            print(f"\n⚠️ El reel {reel_url} ha fallat (estat: {status}).")
            print("🔄 Activant fallback immediat: saltant automàticament al següent candidat...")

    print(f"\n❌ S'han esgotat els {MAX_RETRIES} intents consecutius sense èxit.")


if __name__ == "__main__":
    main()
