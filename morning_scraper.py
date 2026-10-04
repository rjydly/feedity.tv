import os
import re
import csv
import json
import random
from apify_client import ApifyClient

# --- CONFIGURACIÓ ---
APIFY_TOKEN = os.getenv('APIFY_TOKEN') or os.getenv('APIFY_API_TOKEN')

ACCOUNTS_FILE = 'accounts.csv'
BACKUP_CSV = 'backup_reels.csv'
TODAY_QUEUE_FILE = 'today_queue.json'
DB_FILE = 'processed_videos.json'

REELS_PER_ACCOUNT = 5
NUM_RANDOM_ACCOUNTS = 10


def extract_shortcode(url):
    """Mateixa lògica que publisher_runner: /p/X/, /reel/X/, /reels/X/, /tv/X/ -> X."""
    match = re.search(r"instagram\.com/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)", url or "")
    return match.group(1) if match else (url or "").strip()


def load_blocked_shortcodes():
    """Shortcodes que ja han estat publicats ('done') o han fallat ('failed') a backup_reels.csv.
    No s'han de tornar a encuar encara que continuïn entre els últims posts del compte."""
    blocked = set()
    for r in load_backup_csv():
        if str(r.get('status', '')).strip().lower() in ('done', 'failed'):
            blocked.add(extract_shortcode(r.get('link', '')))
    return blocked


def load_processed_ids():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def load_accounts():
    """Llegeix TOTS els comptes de accounts.csv i en tria 10 de forma aleatòria."""
    if not os.path.exists(ACCOUNTS_FILE):
        print(f"⚠️ Fitxer {ACCOUNTS_FILE} no trobat.")
        return []

    urls = []
    with open(ACCOUNTS_FILE, mode='r', encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip()
            if clean_line and not clean_line.startswith('instagram_handle'):
                if not clean_line.startswith('http'):
                    clean_line = f"https://www.instagram.com/{clean_line.replace('@', '')}/"
                urls.append(clean_line)

    if not urls:
        return []

    selected = random.sample(urls, min(NUM_RANDOM_ACCOUNTS, len(urls)))
    print(f"🎲 S'han seleccionat {len(selected)} comptes de forma aleatòria de {len(urls)} disponibles:")
    for u in selected:
        print(f"   • {u}")
    return selected


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
    """Guarda el CSV ordenant sempre de MÉS VIRAL a MENYS VIRAL per likes."""
    for r in rows:
        try:
            r['likes'] = int(r.get('likes', 0))
        except (ValueError, TypeError):
            r['likes'] = 0

    rows.sort(key=lambda x: x['likes'], reverse=True)

    fieldnames = ['link', 'likes', 'status']
    with open(BACKUP_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def sync_candidates_to_backup_csv(candidates):
    """Afegeix els candidats nous trobats per Apify al CSV de backup."""
    rows = load_backup_csv()
    existing_codes = {extract_shortcode(r['link']) for r in rows}

    added_count = 0
    for c in candidates:
        link = c.get('url') or f"https://www.instagram.com/p/{c.get('code') or c.get('shortCode')}/"
        code = extract_shortcode(link)
        if link and code not in existing_codes:
            likes = c.get('likesCount', 0)
            rows.append({
                'link': link,
                'likes': likes,
                'status': ''  # Pendent
            })
            existing_codes.add(code)
            added_count += 1

    save_backup_csv(rows)
    print(f"📦 Sync amb {BACKUP_CSV}: Afegits {added_count} Reels nous al backup. Total al backup: {len(rows)}")


def is_valid_video(item):
    """
    Verifica si l'element d'Apify és un VÍDEO/REEL descarregable.

    IMPORTANT: productType 'feed' NO vol dir vídeo: és com Instagram etiqueta els posts
    normals (imatges i carrusels). Acceptar-lo feia entrar a la cua posts sense vídeo,
    que yt-dlp rebutja amb "There is no video in this post".
    """
    item_type = str(item.get("type") or "").lower()
    product_type = str(item.get("productType") or "").lower()

    # Reel de veritat
    if product_type == "clips":
        return True

    # Imatges i carrusels (sense ser reel) fora
    if item_type in ("image", "sidecar", "carousel"):
        return False

    # Indicis explícits de vídeo
    if item.get("isVideo") or item.get("videoUrl"):
        return True
    if item_type in ("video", "reel"):
        return True
    if item.get("videoViewCount") or item.get("videoPlayCount"):
        return True
    return False


def main():
    if not APIFY_TOKEN:
        print("❌ Error: APIFY_TOKEN / APIFY_API_TOKEN no està configurat.")
        return

    accounts_to_scrape = load_accounts()
    if not accounts_to_scrape:
        print("❌ No s'han trobat comptes a accounts.csv")
        return

    client = ApifyClient(APIFY_TOKEN)
    processed_ids = load_processed_ids()

    print(f"🔍 Executant Apify scraper per a {len(accounts_to_scrape)} comptes...")

    run_input = {
        "directUrls": accounts_to_scrape,
        "resultsType": "posts",
        "resultsLimit": REELS_PER_ACCOUNT
    }

    try:
        run = client.actor("apify/instagram-api-scraper").call(run_input=run_input)
        if not run:
            print("❌ L'actor d'Apify ha fallat o ha retornat None.")
            return

        dataset_id = getattr(run, "default_dataset_id", None) or (run.get("defaultDatasetId") if isinstance(run, dict) else None)
        if not dataset_id:
            print("❌ No s'ha trobat dataset_id a la resposta d'Apify.")
            return

        items = list(client.dataset(dataset_id).iterate_items())
        print(f"✅ Apify ha retornat {len(items)} publicacions/reels.")
    except Exception as e:
        print(f"❌ Error durant la crida a Apify: {e}")
        return

    blocked = load_blocked_shortcodes()
    candidates = []
    skipped_not_video = 0
    skipped_known = 0
    for i in items:
        if not is_valid_video(i):
            skipped_not_video += 1
            continue

        item_id = str(i.get("id") or i.get("shortCode") or i.get("code") or "")
        shortcode = i.get("shortCode") or i.get("code") or item_id

        # Ja publicat, o ja ha fallat anteriorment (post esborrat, sense vídeo...)
        if (item_id in processed_ids or shortcode in processed_ids
                or item_id in blocked or shortcode in blocked):
            skipped_known += 1
            continue

        candidates.append(i)

    print(f"🚫 Descartats {skipped_not_video} posts que no són vídeo (imatges/carrusels) i "
          f"{skipped_known} ja publicats o fallats.")
    print(f"📊 S'han trobat {len(candidates)} Reels candidats nous (no processats anteriorment).")

    if not candidates:
        print("ℹ️ No s'ha trobat cap Reel nou (tots els extrets ja s'havien processat anteriorment).")
        return

    # 1. Sincronitzar TOTS els candidats nous al CSV de backup
    sync_candidates_to_backup_csv(candidates)

    # 2. Ordenar els candidats per likes de MÉS a MENYS
    candidates.sort(key=lambda x: x.get("likesCount", 0), reverse=True)

    # 3. Agafar els 3 MILLORS Reels per a la cua d'avui
    today_top_3 = []
    for c in candidates[:3]:
        shortcode = c.get("shortCode") or c.get("code") or c.get("id")
        link = c.get('url') or f"https://www.instagram.com/reel/{shortcode}/"
        item_id = str(c.get("id") or shortcode or "")
        likes = c.get("likesCount", 0)
        today_top_3.append({
            "id": item_id,
            "url": link,
            "likes": likes
        })

    with open(TODAY_QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(today_top_3, f, indent=4)

    print(f"🎉 [CUA D'AVUI GENERADA] {len(today_top_3)} Reels guardats a {TODAY_QUEUE_FILE}:")
    for reel in today_top_3:
        print(f"   🔥 {reel['url']} ({reel['likes']} likes)")


if __name__ == "__main__":
    main()