import os
import re
import csv
import json
import glob
import html
import time
import base64
import subprocess
from io import BytesIO
from datetime import datetime, timezone
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import yt_dlp
from moviepy import VideoFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips

# ==========================================
# CONFIGURACIÓ PRINCIPAL
# ==========================================

TEST_MODE = True

# Secrets i credencials
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
INSTAGRAM_COOKIES_FILE = os.getenv("INSTAGRAM_COOKIES_FILE")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BUFFER_ACCESS_TOKEN = os.getenv("BUFFER_ACCESS_TOKEN")
BUFFER_CHANNEL_IDS = os.getenv("BUFFER_CHANNEL_IDS")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")

# Bundle.social (Snapchat Spotlight)
BUNDLE_SOCIAL_API_KEY = os.getenv("BUNDLE_SOCIAL_API_KEY")
BUNDLE_TEAM_ID = os.getenv("BUNDLE_TEAM_ID")
SNAPCHAT_TRACKER_FILE = "snapchat_tracker.json"
SNAPCHAT_MAX_MONTHLY = 20
SNAPCHAT_MIN_LIKES = int(os.getenv("SNAPCHAT_MIN_LIKES", "10000"))

# Arxius de persistència i seguiment
PUBLISHED_TRACKING_CSV = "published_posts.csv"

# Rutes de recursos i carpetes
ASSETS_DIR = "assets"
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
VIDEOS_DIR = "videos"

# Estil i colors
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (227, 177, 0)      # Groc corporatiu #e3b100
COLOR_MUTED = (113, 118, 123)

# Text de drets obligatori al final del caption
DISCLAIMER_TEXT = "All rights belong to the respective owner. DM for credit or removal."


# ==========================================
# GESTIÓ D'HISTORIAL I CSVs
# ==========================================

def extract_shortcode(reel_url):
    match = re.search(r"instagram\.com/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)", reel_url or "")
    return match.group(1) if match else (reel_url or "").strip()


def load_processed_ids():
    if os.path.exists("processed_videos.json"):
        try:
            with open("processed_videos.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_processed_id(video_id, shortcode=None):
    if TEST_MODE:
        print("ℹ️ TEST_MODE actiu: No es desa l'ID a processed_videos.json")
        return
    history = load_processed_ids()
    changed = False
    for vid in [video_id, shortcode]:
        if vid and str(vid) not in history:
            history.append(str(vid))
            changed = True
    if changed:
        with open("processed_videos.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)


def update_csv_status(target_url, new_status="done"):
    """Actualitza l'estat tant a sources.csv com a backup_reels.csv."""
    if TEST_MODE:
        return

    target_shortcode = extract_shortcode(target_url)

    # 1. Actualitzar sources.csv
    if os.path.exists("sources.csv"):
        rows = []
        with open("sources.csv", mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0].strip() == target_url.strip() or extract_shortcode(row[0].strip()) == target_shortcode:
                    rows.append([row[0].strip(), new_status])
                else:
                    rows.append(row)

        with open("sources.csv", mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"📝 sources.csv actualitzat: {target_url} -> {new_status}")

    # 2. Actualitzar backup_reels.csv
    if os.path.exists("backup_reels.csv"):
        backup_rows = []
        updated_backup = False
        with open("backup_reels.csv", mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                link = row.get("link", "").strip()
                if link == target_url.strip() or extract_shortcode(link) == target_shortcode:
                    row["status"] = new_status
                    updated_backup = True
                backup_rows.append(row)

        if updated_backup:
            with open("backup_reels.csv", mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["link", "likes", "status"])
                writer.writeheader()
                writer.writerows(backup_rows)
            print(f"📝 backup_reels.csv actualitzat: {target_url} -> {new_status}")


def record_published_post_tracking(tracking_id, shortcode, source_url, source_account, initial_likes, thumbnail_title, snapchat_published=False):
    """Guarda l'ID i les mètriques d'origen a published_posts.csv per a traçabilitat."""
    if TEST_MODE:
        return

    file_exists = os.path.exists(PUBLISHED_TRACKING_CSV)
    fieldnames = [
        "id",
        "shortcode",
        "source_url",
        "source_account",
        "initial_likes",
        "thumbnail_title",
        "published_date",
        "snapchat_published"
    ]
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "id": tracking_id,
        "shortcode": shortcode,
        "source_url": source_url,
        "source_account": source_account,
        "initial_likes": initial_likes,
        "thumbnail_title": thumbnail_title,
        "published_date": now_iso,
        "snapchat_published": "yes" if snapchat_published else "no"
    }

    with open(PUBLISHED_TRACKING_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"📊 Registre guardat a {PUBLISHED_TRACKING_CSV} per al ID {tracking_id}")


# ==========================================
# SNAPCHAT TRACKER (BUNDLE.SOCIAL)
# ==========================================

def get_reel_likes_count(target_url):
    """Cerca els likes del Reel a backup_reels.csv o today_queue.json."""
    shortcode = extract_shortcode(target_url)

    if os.path.exists("today_queue.json"):
        try:
            with open("today_queue.json", "r", encoding="utf-8") as f:
                queue = json.load(f)
                for item in queue:
                    if extract_shortcode(item.get("url", "")) == shortcode:
                        return int(item.get("likes", 0))
        except Exception:
            pass

    if os.path.exists("backup_reels.csv"):
        try:
            with open("backup_reels.csv", "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if extract_shortcode(row.get("link", "")) == shortcode:
                        return int(row.get("likes", 0))
        except Exception:
            pass

    return 0


def load_snapchat_tracker():
    now = datetime.now(timezone.utc)
    current_month = now.strftime("%Y-%m")

    default_data = {
        "month": current_month,
        "count": 0,
        "last_post_date": ""
    }

    if not os.path.exists(SNAPCHAT_TRACKER_FILE):
        return default_data

    try:
        with open(SNAPCHAT_TRACKER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("month") != current_month:
                data["month"] = current_month
                data["count"] = 0
            return data
    except Exception:
        return default_data


def save_snapchat_tracker(tracker_data):
    with open(SNAPCHAT_TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(tracker_data, f, indent=4)


def should_publish_to_snapchat(reel_url, video_path):
    if not BUNDLE_SOCIAL_API_KEY or not BUNDLE_TEAM_ID:
        return False

    tracker = load_snapchat_tracker()
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if tracker.get("count", 0) >= SNAPCHAT_MAX_MONTHLY:
        print(f"🛑 [Snapchat] Quota mensual esgotada ({tracker['count']}/{SNAPCHAT_MAX_MONTHLY}). S'omet.")
        return False

    if tracker.get("last_post_date") == today_str:
        print("⏩ [Snapchat] Ja s'ha publicat el vídeo Top d'avui a Snapchat. S'omet.")
        return False

    likes = get_reel_likes_count(reel_url)
    if likes < SNAPCHAT_MIN_LIKES:
        print(f"ℹ️ [Snapchat] El vídeo té {likes} likes (mínim requerit: {SNAPCHAT_MIN_LIKES}). S'omet.")
        return False

    try:
        clip = VideoFileClip(video_path)
        dur = clip.duration
        clip.close()
        if dur < 5.0 or dur > 180.0:
            print(f"⚠️ [Snapchat] Durada de {dur:.1f}s fora del rang permès (5-180s).")
            return False
    except Exception as e:
        print(f"⚠️ [Snapchat] Error verificant durada: {e}")

    print(f"🌟 [Snapchat] Candidat VIP acceptat! ({likes} likes. Quota del mes: {tracker['count'] + 1}/{SNAPCHAT_MAX_MONTHLY})")
    return True


def record_snapchat_publication():
    tracker = load_snapchat_tracker()
    tracker["count"] = tracker.get("count", 0) + 1
    tracker["last_post_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    save_snapchat_tracker(tracker)


def publish_to_bundle_snapchat(video_path, thumbnail_title, short_hook, tracking_id):
    if not BUNDLE_SOCIAL_API_KEY or not BUNDLE_TEAM_ID:
        return False

    headers = {"x-api-key": BUNDLE_SOCIAL_API_KEY}

    print("📤 [Bundle.social] Pujant vídeo per a Snapchat Spotlight...")
    try:
        with open(video_path, "rb") as f:
            upload_res = requests.post(
                "https://api.bundle.social/api/v1/upload",
                headers=headers,
                files={"file": f},
                data={"teamId": BUNDLE_TEAM_ID},
                timeout=60
            )

        if upload_res.status_code not in (200, 201):
            print(f"❌ Error en upload de Bundle.social ({upload_res.status_code}): {upload_res.text}")
            return False

        upload_data = upload_res.json()
        upload_id = upload_data.get("uploadId") or upload_data.get("id")
        if not upload_id:
            return False

    except Exception as e:
        print(f"❌ Error connectant amb upload de Bundle.social: {e}")
        return False

    clean_hook = (short_hook or thumbnail_title or "").replace("**", "").replace("\n", " ").strip()
    id_tag = f"#{tracking_id}"
    max_text_len = 160 - len(id_tag) - 2

    if len(clean_hook) > max_text_len:
        snap_desc = f"{clean_hook[:max_text_len-3]}... {id_tag}"
    else:
        snap_desc = f"{clean_hook} {id_tag}"

    now_iso = datetime.now(timezone.utc).isoformat()

    post_payload = {
        "teamId": BUNDLE_TEAM_ID,
        "title": thumbnail_title[:40],
        "status": "SCHEDULED",
        "postDate": now_iso,
        "socialAccountTypes": ["SNAPCHAT"],
        "data": {
            "SNAPCHAT": {
                "type": "SPOTLIGHT",
                "uploadIds": [upload_id],
                "description": snap_desc[:160],
                "locale": "en_US",
                "skipSaveToProfile": False
            }
        }
    }

    try:
        print("🚀 [Bundle.social] Creant post a Snapchat Spotlight...")
        post_res = requests.post(
            "https://api.bundle.social/api/v1/post",
            headers={**headers, "Content-Type": "application/json"},
            json=post_payload,
            timeout=30
        )
        if post_res.status_code in (200, 201):
            print("🎉 Publicat amb èxit a Snapchat Spotlight!")
            return True
        else:
            print(f"❌ Error a Snapchat Spotlight ({post_res.status_code}): {post_res.text}")
            return False
    except Exception as e:
        print(f"❌ Error cridant a Bundle.social: {e}")
        return False


# ==========================================
# GESTIÓ DE MEDIA I COMMIT A GITHUB
# ==========================================

def cleanup_videos_dir(keep_filenames=None):
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    keep_filenames = keep_filenames or []
    for f in os.listdir(VIDEOS_DIR):
        if f not in keep_filenames:
            full_path = os.path.join(VIDEOS_DIR, f)
            try:
                if os.path.isfile(full_path):
                    os.remove(full_path)
            except OSError:
                pass


def push_media_to_github(keep_filenames):
    if TEST_MODE:
        return True

    print("📤 Netejant fitxers antics i sincronitzant commit a GitHub...")
    try:
        cleanup_videos_dir(keep_filenames=keep_filenames)

        subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=False)

        subprocess.run([
            "git", "add", "-A",
            VIDEOS_DIR,
            "processed_videos.json",
            "sources.csv",
            "today_queue.json",
            "backup_reels.csv",
            SNAPCHAT_TRACKER_FILE,
            PUBLISHED_TRACKING_CSV
        ], check=False)

        subprocess.run(["git", "commit", "-m", "Publish dual videos & update tracking [skip ci]"], check=False)
        subprocess.run(["git", "push"], check=True)
        print("✅ Estat i vídeos sincronitzats a GitHub amb èxit!")
        time.sleep(3)
        return True
    except Exception as e:
        print(f"⚠️ Error fent push a GitHub: {e}")
        return False


# ==========================================
# PUBLICACIÓ VIA BUFFER GRAPHQL API
# ==========================================

def get_channel_service(channel_id, headers):
    query = """
    query GetChannel($input: ChannelInput!) {
      channel(input: $input) {
        id
        service
      }
    }
    """
    try:
        res = requests.post(
            "https://api.buffer.com",
            headers=headers,
            json={"query": query, "variables": {"input": {"id": channel_id}}},
            timeout=15
        )
        data = res.json()
        ch = (data.get("data") or {}).get("channel")
        if ch and "service" in ch:
            return str(ch["service"]).lower()
    except Exception as e:
        print(f"ℹ️ Consulta canal {channel_id}: {e}")
    return ""


def publish_to_buffer(caption_text, fb_video_filename, shorts_video_filename, thumbnail_offset_ms=0):
    if not BUFFER_ACCESS_TOKEN or not BUFFER_CHANNEL_IDS or not GITHUB_REPOSITORY:
        print("⚠️ Dades de Buffer o GITHUB_REPOSITORY no configurades. S'omet la publicació.")
        return False

    channel_list = [c.strip() for c in BUFFER_CHANNEL_IDS.split(",") if c.strip()]
    if not channel_list:
        print("⚠️ No hi ha cap channel_id vàlid a BUFFER_CHANNEL_IDS.")
        return False

    headers = {
        "Authorization": f"Bearer {BUFFER_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    mutation = """
    mutation CreatePost($input: CreatePostInput!) {
      createPost(input: $input) {
        ... on PostActionSuccess {
          post {
            id
            status
          }
        }
        ... on MutationError {
          message
        }
      }
    }
    """

    all_success = True
    for channel_id in channel_list:
        service = get_channel_service(channel_id, headers)

        # Facebook rep la versió llarga; Instagram i TikTok reben la versió Shorts
        if "facebook" in service:
            chosen_video_file = fb_video_filename
            print(f"📘 Canal Facebook detectat ({channel_id}): enviant versió amb text llarg ({chosen_video_file})")
        else:
            chosen_video_file = shorts_video_filename
            print(f"📱 Canal Shorts detectat ({service.upper() or 'IG/TIKTOK'} - {channel_id}): enviant versió amb text gran ({chosen_video_file})")

        public_video_url = f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/main/{VIDEOS_DIR}/{chosen_video_file}"

        post_input = {
            "channelId": channel_id,
            "text": caption_text,
            "schedulingType": "automatic",
            "mode": "shareNow",
            "assets": [
                {
                    "video": {
                        "url": public_video_url,
                        "metadata": {
                            "thumbnailOffset": thumbnail_offset_ms
                        }
                    }
                }
            ]
        }

        if "instagram" in service:
            post_input["metadata"] = {"instagram": {"type": "reel", "shouldShareToFeed": True}}
        elif "facebook" in service:
            post_input["metadata"] = {"facebook": {"type": "reel"}}

        def send_request(inp):
            return requests.post(
                "https://api.buffer.com",
                headers=headers,
                json={"query": mutation, "variables": {"input": inp}},
                timeout=30
            )

        try:
            response = send_request(post_input)
            res_data = response.json()
            result = (res_data.get("data") or {}).get("createPost", {})
            error_msg = result.get("message") or ""

            if "Instagram posts require a type" in error_msg:
                post_input["metadata"] = {"instagram": {"type": "reel", "shouldShareToFeed": True}}
                response = send_request(post_input)
                res_data = response.json()
                result = (res_data.get("data") or {}).get("createPost", {})
                error_msg = result.get("message") or ""
            elif "Facebook posts require a type" in error_msg:
                post_input["metadata"] = {"facebook": {"type": "reel"}}
                response = send_request(post_input)
                res_data = response.json()
                result = (res_data.get("data") or {}).get("createPost", {})
                error_msg = result.get("message") or ""

            if "errors" in res_data:
                print(f"❌ Error GraphQL al canal {channel_id}: {json.dumps(res_data['errors'], indent=2)}")
                all_success = False
            elif error_msg:
                print(f"⚠️ Resposta de Buffer al canal {channel_id}: {error_msg}")
                all_success = False
            elif "post" in result:
                print(f"🎉 Publicat amb èxit al canal {service.upper() or channel_id}! Post ID: {result['post']['id']}")

        except Exception as e:
            print(f"❌ Error connectant amb Buffer ({channel_id}): {e}")
            all_success = False

    return all_success


# ==========================================
# TIPOGRAFIES I RENDERITZAT (PLUS JAKARTA SANS)
# ==========================================

def ensure_fonts():
    os.makedirs(FONTS_DIR, exist_ok=True)
    font_urls = {
        "PlusJakartaSans-Regular.ttf": "https://raw.githubusercontent.com/tokotype/PlusJakartaSans/master/fonts/ttf/PlusJakartaSans-Regular.ttf",
        "PlusJakartaSans-Bold.ttf": "https://raw.githubusercontent.com/tokotype/PlusJakartaSans/master/fonts/ttf/PlusJakartaSans-Bold.ttf",
        "PlusJakartaSans-SemiBold.ttf": "https://raw.githubusercontent.com/tokotype/PlusJakartaSans/master/fonts/ttf/PlusJakartaSans-SemiBold.ttf"
    }

    for font_file, url in font_urls.items():
        dest = os.path.join(FONTS_DIR, font_file)
        if not os.path.exists(dest):
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with open(dest, "wb") as f:
                        f.write(r.content)
            except Exception as e:
                print(f"⚠️ Error descarregant {font_file}: {e}")


def get_jakarta_font(style="regular", size=42):
    ensure_fonts()
    font_map = {
        "bold": os.path.join(FONTS_DIR, "PlusJakartaSans-Bold.ttf"),
        "semibold": os.path.join(FONTS_DIR, "PlusJakartaSans-SemiBold.ttf"),
        "regular": os.path.join(FONTS_DIR, "PlusJakartaSans-Regular.ttf")
    }

    path = font_map.get(style, font_map["regular"])
    if os.path.exists(path):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass

    for fallback in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(fallback):
            return ImageFont.truetype(fallback, size=size)

    return ImageFont.load_default()


def clean_text_symbols(text):
    if not text:
        return ""
    emoji_pattern = re.compile(
        "["
        "\U00010000-\U0010ffff"
        "\u200d\u200c\u200e\u200f"
        "\u2300-\u23ff"
        "\u2600-\u27bf"
        "\u2190-\u21ff"
        "\u2200-\u22ff"
        "\u2b50\u2b06\u2b07"
        "]+",
        flags=re.UNICODE
    )
    cleaned = emoji_pattern.sub("", text)
    cleaned = cleaned.replace("≡", "").replace("■", "").replace("□", "")
    return cleaned.strip()


def tokenize_markdown_text(text):
    paragraphs = text.split("\n")
    tokenized_paragraphs = []

    for para in paragraphs:
        if not para.strip():
            tokenized_paragraphs.append([])
            continue

        parts = re.split(r'(\*\*[^*]+\*\*)', para)
        tokens = []
        for part in parts:
            if not part:
                continue
            if part.startswith("**") and part.endswith("**") and len(part) >= 4:
                bold_words = part[2:-2].split()
                for w in bold_words:
                    tokens.append((w, True))
            else:
                regular_words = part.split()
                for w in regular_words:
                    tokens.append((w, False))
        tokenized_paragraphs.append(tokens)

    return tokenized_paragraphs


def wrap_tokenized_text(tokenized_paragraphs, regular_font, bold_font, max_width, draw):
    all_lines = []
    space_w_reg = draw.textbbox((0, 0), " ", font=regular_font)[2]

    for para_tokens in tokenized_paragraphs:
        if not para_tokens:
            all_lines.append([])
            continue

        current_line = []
        current_width = 0

        for word, is_bold in para_tokens:
            f = bold_font if is_bold else regular_font
            word_w = draw.textbbox((0, 0), word, font=f)[2]

            needed_width = word_w if not current_line else (current_width + space_w_reg + word_w)

            if needed_width <= max_width:
                current_line.append((word, is_bold, word_w))
                current_width = needed_width
            else:
                if current_line:
                    all_lines.append(current_line)
                    current_line = [(word, is_bold, word_w)]
                    current_width = word_w
                else:
                    all_lines.append([(word, is_bold, word_w)])
                    current_line = []
                    current_width = 0

        if current_line:
            all_lines.append(current_line)

    return all_lines


# -------------------------------------------------------------
# CAPÇALERA 1: FACEBOOK (Estil Tweet amb text explicatiu llarg)
# -------------------------------------------------------------
def create_facebook_header_image(facebook_text, width=1080):
    margin_x = 75
    max_text_width = width - (margin_x * 2)

    name_font = get_jakarta_font("semibold", size=48)
    handle_font = get_jakarta_font("regular", size=38)
    body_font_regular = get_jakarta_font("regular", size=44)
    body_font_bold = get_jakarta_font("bold", size=44)

    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    tokenized = tokenize_markdown_text(facebook_text)
    wrapped_lines = wrap_tokenized_text(tokenized, body_font_regular, body_font_bold, max_text_width, dummy_draw)

    line_height = 62
    paragraph_gap = 26

    body_height = 0
    for line in wrapped_lines:
        if not line:
            body_height += paragraph_gap
        else:
            body_height += line_height

    avatar_size = 110
    top_padding = 50
    bottom_padding = 40
    header_height = top_padding + avatar_size + 36 + body_height + bottom_padding

    img = Image.new("RGBA", (width, header_height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    avatar_x = margin_x
    avatar_y = top_padding

    logo_file = LOGO_PATH if os.path.exists(LOGO_PATH) else ("logo.png" if os.path.exists("logo.png") else None)
    if logo_file:
        try:
            logo_img = Image.open(logo_file).convert("RGBA").resize((avatar_size, avatar_size), Image.Resampling.LANCZOS)
            mask = Image.new("L", (avatar_size, avatar_size), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, avatar_size, avatar_size), fill=255)
            img.paste(logo_img, (avatar_x, avatar_y), mask)
        except Exception:
            draw.ellipse([avatar_x, avatar_y, avatar_x + avatar_size, avatar_y + avatar_size], fill=(22, 24, 28))
            draw.text((avatar_x + 32, avatar_y + 20), "F", font=name_font, fill=COLOR_YELLOW)
    else:
        draw.ellipse([avatar_x, avatar_y, avatar_x + avatar_size, avatar_y + avatar_size], fill=(22, 24, 28))
        draw.text((avatar_x + 32, avatar_y + 20), "F", font=name_font, fill=COLOR_YELLOW)

    text_start_x = avatar_x + avatar_size + 24
    draw.text((text_start_x, avatar_y + 6), "Feedity", font=name_font, fill=COLOR_WHITE)
    draw.text((text_start_x, avatar_y + 58), "@feedity.tv", font=handle_font, fill=COLOR_MUTED)

    text_y = avatar_y + avatar_size + 36
    space_w = dummy_draw.textbbox((0, 0), " ", font=body_font_regular)[2]

    for line in wrapped_lines:
        if not line:
            text_y += paragraph_gap
            continue

        cursor_x = margin_x
        for word, is_bold, word_w in line:
            f = body_font_bold if is_bold else body_font_regular
            # Paraula en negreta destacada en groc #e3b100
            color = COLOR_YELLOW if is_bold else COLOR_WHITE
            draw.text((cursor_x, text_y), word, font=f, fill=color)
            cursor_x += word_w + space_w

        text_y += line_height

    return np.array(img)


# -------------------------------------------------------------
# CAPÇALERA 2: SHORTS (Logo a sobre i text curt en gran)
# -------------------------------------------------------------
def create_shorts_header_image(short_hook, width=1080):
    margin_x = 80
    max_text_width = width - (margin_x * 2)

    body_font_regular = get_jakarta_font("semibold", size=52)
    body_font_bold = get_jakarta_font("bold", size=54)

    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    tokenized = tokenize_markdown_text(short_hook)
    wrapped_lines = wrap_tokenized_text(tokenized, body_font_regular, body_font_bold, max_text_width, dummy_draw)

    # Limitem estrictament a màxim 2 línies per a Shorts
    wrapped_lines = [l for l in wrapped_lines if l][:2]

    line_height = 72
    body_height = len(wrapped_lines) * line_height

    logo_size = 96
    top_padding = 40
    gap_logo_text = 24
    bottom_padding = 32
    header_height = top_padding + logo_size + gap_logo_text + body_height + bottom_padding

    img = Image.new("RGBA", (width, header_height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    # Logo centrat a sobre del text
    logo_x = (width - logo_size) // 2
    logo_y = top_padding

    logo_file = LOGO_PATH if os.path.exists(LOGO_PATH) else ("logo.png" if os.path.exists("logo.png") else None)
    if logo_file:
        try:
            logo_img = Image.open(logo_file).convert("RGBA").resize((logo_size, logo_size), Image.Resampling.LANCZOS)
            mask = Image.new("L", (logo_size, logo_size), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, logo_size, logo_size), fill=255)
            img.paste(logo_img, (logo_x, logo_y), mask)
        except Exception:
            draw.ellipse([logo_x, logo_y, logo_x + logo_size, logo_y + logo_size], fill=(22, 24, 28))
            f_font = get_jakarta_font("bold", size=52)
            draw.text((logo_x + 30, logo_y + 14), "F", font=f_font, fill=COLOR_YELLOW)
    else:
        draw.ellipse([logo_x, logo_y, logo_x + logo_size, logo_y + logo_size], fill=(22, 24, 28))
        f_font = get_jakarta_font("bold", size=52)
        draw.text((logo_x + 30, logo_y + 14), "F", font=f_font, fill=COLOR_YELLOW)

    text_y = logo_y + logo_size + gap_logo_text
    space_w = dummy_draw.textbbox((0, 0), " ", font=body_font_regular)[2]

    # Dibuixem cada línia centrada
    for line in wrapped_lines:
        line_w = sum(w for _, _, w in line) + (len(line) - 1) * space_w
        cursor_x = (width - line_w) // 2

        for word, is_bold, word_w in line:
            f = body_font_bold if is_bold else body_font_regular
            # Paraula destacada en groc #e3b100
            color = COLOR_YELLOW if is_bold else COLOR_WHITE
            draw.text((cursor_x, text_y), word, font=f, fill=color)
            cursor_x += word_w + space_w

        text_y += line_height

    return np.array(img)


# ==========================================
# INTEL·LIGÈNCIA ARTIFICIAL (GEMINI & GROQ)
# ==========================================

AI_PROMPT_INSTRUCTIONS = f"""
Examine this video frame and the original post description carefully.

CRITICAL CONSISTENCY RULE:
Both 'facebook_text', 'short_hook' and 'generated_caption' MUST be 100% focused on the EXACT SAME topic shown in the video.
- If the video is a meme, funny moment, or comedy, explain that specific funny situation. NEVER invent an unrelated scientific, historical, or geographical fact!
- If the video is a scientific discovery or educational fact, explain that specific discovery.

RULES FOR CREDITS:
1. Identify the TRUE ORIGINAL source/creator of the video (e.g. if description says "Media: @creator", "Video by @author", or shows primary watermark, credit is @author).
2. NEVER credit the reposter / curator aggregator (e.g. ignore @wealth, @pubity, etc.).
3. If no third-party source is mentioned, set "credits" to "".

RULES FOR FACEBOOK TEXT ('facebook_text'):
- Paraphrase the video message into a clean, viral narrative in ENGLISH in 1 or 2 short paragraphs (separated by \\n\\n).
- STRICTLY NO EMOJIS OR UNICODE SYMBOLS in 'facebook_text'.
- EMPHASIZE 2-4 key punchline words using markdown asterisks **like this**.

RULES FOR SHORT HOOK ('short_hook') [FOR IG/TIKTOK/SNAPCHAT]:
- Ultra-short, high-impact punchy hook in ENGLISH in strictly 1 OR 2 LINES (6 to 12 words total).
- Designed for fast-scrolling vertical viewers.
- STRICTLY NO EMOJIS OR UNICODE SYMBOLS.
- EMPHASIZE 1-3 key punchline words using markdown asterisks **like this**.

RULES FOR THUMBNAIL TITLE ('thumbnail_title'):
- Ultra-punchy, high-impact headline of 3 TO 6 WORDS in UPPERCASE ENGLISH directly about the video content.
- Example: "GOOGLE CONFUSED BY GOOGLE" or "THE REAL NIGHT SKY".

RULES FOR THE CAPTION ('generated_caption'):
Structure in this exact order:
1. Engaging Hook & detailed explanation directly about the video subject (context, why it is funny or amazing).
2. Call to Action (CTA) (e.g. 'Have you ever tried this? Tell us below! 👇').
3. 8-12 targeted viral hashtags relevant to this specific topic.
4. Credit line (ONLY if true original source identified):
   Credit: @original_author

Return strictly a JSON object with this format:
{{
  "credits": "@original_creator_or_empty",
  "facebook_text": "First line hook\\n\\nSecond line with **bold words**.",
  "short_hook": "When you try your **hardest** and still **fail**.",
  "thumbnail_title": "PUNCHY HEADLINE HERE",
  "generated_caption": "Detailed story directly about this video...\\n\\nCTA\\n\\n#hashtags\\n\\nCredit: @original_author"
}}
"""


def format_final_caption(generated_caption, tracking_id):
    caption = (generated_caption or "").strip()
    id_tag = f"ID: #{tracking_id}"
    if id_tag not in caption:
        caption = f"{caption}\n\n{id_tag}" if caption else id_tag
    if DISCLAIMER_TEXT not in caption:
        caption = f"{caption}\n\n{DISCLAIMER_TEXT}"
    return caption


def parse_json_safely(raw_text):
    try:
        return json.loads(raw_text)
    except Exception:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            return json.loads(match.group())
    return None


def extract_frame_as_image(video_path, timestamp=0.5):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame = cap.read()
    cap.release()
    if success and frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None


def image_to_base64_jpeg(image_pil, max_dim=1024):
    img = image_pil.copy()
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=80)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def analyze_with_gemini_vision(image_pil, caption_raw="", tracking_id=""):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    contents = [AI_PROMPT_INSTRUCTIONS]
    if image_pil is not None:
        contents.append(image_pil)
    if caption_raw:
        contents.append(f"\nOriginal post description: {caption_raw}")

    candidate_models = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite"
    ]

    for model_name in candidate_models:
        try:
            print(f"🧠 [Gemini] Provant model {model_name}...")
            res = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7
                )
            )
            data = parse_json_safely(res.text)
            if data:
                fb_text = clean_text_symbols(data.get("facebook_text") or data.get("tweet_text", ""))
                short_hook = clean_text_symbols(data.get("short_hook") or fb_text.split("\n")[0])
                if fb_text:
                    return (
                        data.get("credits", ""),
                        fb_text,
                        short_hook,
                        format_final_caption(data.get("generated_caption", ""), tracking_id),
                        data.get("thumbnail_title", "FEATURED STORY").upper()
                    )
        except Exception as e:
            print(f"ℹ️ Gemini error ({model_name}): {e}")
            time.sleep(1)
            continue
    return None


def analyze_with_groq_vision(image_pil, caption_raw="", tracking_id=""):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    prompt = AI_PROMPT_INSTRUCTIONS
    if caption_raw:
        prompt += f"\nOriginal post description: {caption_raw}"

    content_multimodal = [{"type": "text", "text": prompt}]
    if image_pil is not None:
        b64 = image_to_base64_jpeg(image_pil, max_dim=1024)
        content_multimodal.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    candidate_models = [
        ("qwen/qwen3.8-27b", True),
        ("openai/gpt-oss-120b", False),
        ("llama-3.3-70b-versatile", False)
    ]

    for model_name, supports_vision in candidate_models:
        try:
            print(f"🧠 [Groq Fallback] Provant model {model_name}...")
            if supports_vision and image_pil is not None:
                messages = [{"role": "user", "content": content_multimodal}]
            else:
                messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.6,
                "max_completion_tokens": 600,
                "response_format": {"type": "json_object"}
            }

            if "qwen" in model_name:
                kwargs["reasoning_effort"] = "none"

            completion = client.chat.completions.create(**kwargs)
            data = parse_json_safely(completion.choices[0].message.content)

            if data:
                fb_text = clean_text_symbols(data.get("facebook_text") or data.get("tweet_text", ""))
                short_hook = clean_text_symbols(data.get("short_hook") or fb_text.split("\n")[0])
                if fb_text:
                    return (
                        data.get("credits", ""),
                        fb_text,
                        short_hook,
                        format_final_caption(data.get("generated_caption", ""), tracking_id),
                        data.get("thumbnail_title", "FEATURED STORY").upper()
                    )
        except Exception as e:
            print(f"ℹ️ Groq error ({model_name}): {e}")
            continue
    return None


def send_telegram_alert(error_detail, reel_url=""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    alert_text = (
        f"🚨 <b>ALERTA CRÍTICA FEEDITY PIPELINE</b> 🚨\n\n"
        f"❌ <b>Error:</b> Cap servei d'Intel·ligència Artificial ha respost després de <b>5 intents</b>.\n\n"
        f"🔗 <b>Reel afectat:</b> {html.escape(reel_url)}\n\n"
        f"📋 <b>Detalls de l'error:</b>\n<code>{html.escape(str(error_detail)[:350])}</code>"
    )
    try:
        requests.post(url_msg, data={"chat_id": TELEGRAM_CHAT_ID, "text": alert_text, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass


def analyze_content_with_retry(image_pil, caption_raw="", reel_url="", tracking_id="", max_retries=5, delay_seconds=60):
    for attempt in range(1, max_retries + 1):
        print(f"\n🤖 [Intent {attempt}/{max_retries}] Analitzant contingut visual i text amb IA...")

        if GEMINI_API_KEY:
            res = analyze_with_gemini_vision(image_pil, caption_raw, tracking_id)
            if res and len(res) == 5:
                return res

        if GROQ_API_KEY:
            res = analyze_with_groq_vision(image_pil, caption_raw, tracking_id)
            if res and len(res) == 5:
                return res

        if attempt < max_retries:
            print(f"⏳ Totes les APIs han fallat o estan saturades. Esperant {delay_seconds} segons...")
            time.sleep(delay_seconds)

    send_telegram_alert("Totes les APIs de visió han fallat.", reel_url)
    return None, None, None, None, None


# ==========================================
# CROP INTEL·LIGENT DE CONTINGUT (ROI PER MOVIMENT)
# ==========================================

CROP_ANALYSIS_WIDTH = 360
CROP_PAIR_GAP_S = 0.5
CROP_NUM_PAIRS = 12
CROP_DIFF_THRESHOLD = 2.5
CROP_ACTIVITY_RATIO = 0.20
CROP_MIN_HEIGHT_RATIO = 0.18
CROP_MIN_AREA_RATIO = 0.12
CROP_MIN_FILL_RATIO = 0.35
CROP_ASPECT_RANGE = (0.5, 2.6)
CROP_INNER_MARGIN = 0.006

CROP_SEED_MIN_AREA_RATIO = 0.015
CROP_EDGE_STEP_MIN = 10
CROP_EDGE_COL_CONSISTENCY = 0.80
CROP_EDGE_FLAT_ROWS = 8
CROP_EDGE_FLAT_STD = 6.0
CROP_EDGE_TEXTURE_STD = 6.0
CROP_EDGE_MOTION_AT_EDGE = 0.30
CROP_EDGE_MIN_FLAT_RUN = 0.06
CROP_EDGE_MAX_GROW_X = 0.5
CROP_EDGE_MAX_GROW_Y = 0.35


def _sample_frame_pairs(clip, num_pairs=CROP_NUM_PAIRS, gap=CROP_PAIR_GAP_S):
    duration = clip.duration or 0
    if duration <= 0.3:
        return []
    gap = min(gap, duration / 3)
    t_max = max(duration - gap - 0.05, 0.1)
    times = np.linspace(0.1, t_max, num=num_pairs)
    pairs = []
    for t in times:
        try:
            a = clip.get_frame(float(t))
            b = clip.get_frame(float(min(t + gap, duration - 0.02)))
            pairs.append((a, b))
        except Exception:
            continue
    return pairs


def _downscale(frame, target_w=CROP_ANALYSIS_WIDTH):
    h, w = frame.shape[:2]
    scale = target_w / float(w)
    small = cv2.resize(frame, (target_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return small, scale


def _validate_box(box, frame_w, frame_h):
    if box is None:
        return False
    x, y, w, h = box
    if w <= 0 or h <= 0 or h < CROP_MIN_HEIGHT_RATIO * frame_h:
        return False
    if (w * h) < CROP_MIN_AREA_RATIO * frame_w * frame_h:
        return False
    ar = w / float(h)
    return CROP_ASPECT_RANGE[0] <= ar <= CROP_ASPECT_RANGE[1]


def find_motion_seed(pairs, frame_w, frame_h):
    if len(pairs) < 3:
        return None, None

    activity = None
    scale = 1.0
    for a, b in pairs:
        sa, scale = _downscale(a)
        sb, _ = _downscale(b)
        diff = np.max(np.abs(sa.astype(np.int16) - sb.astype(np.int16)), axis=2).astype(np.uint8)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        noise_floor = float(np.percentile(diff, 25))
        thr = max(CROP_DIFF_THRESHOLD, noise_floor * 2.0 + 1.5)
        moving = (diff > thr).astype(np.float32)
        activity = moving if activity is None else activity + moving
    activity /= len(pairs)

    mask = (activity >= CROP_ACTIVITY_RATIO).astype(np.uint8)
    raw_mask = mask.copy()

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    pad = 25
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = mask[pad:-pad, pad:-pad]

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _area = stats[best]

    comp = (labels[y:y + h, x:x + w] == best)
    row_frac = comp.mean(axis=1)
    col_frac = comp.mean(axis=0)
    rows = np.where(row_frac >= 0.2)[0]
    cols = np.where(col_frac >= 0.2)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None, None
    y1, y2 = y + rows[0], y + rows[-1] + 1
    x1, x2 = x + cols[0], x + cols[-1] + 1

    fill = mask[y1:y2, x1:x2].mean()
    if fill < CROP_MIN_FILL_RATIO:
        return None, None

    inv = 1.0 / scale
    box = (int(x1 * inv), int(y1 * inv), int((x2 - x1) * inv), int((y2 - y1) * inv))
    if box[2] * box[3] < CROP_SEED_MIN_AREA_RATIO * frame_w * frame_h:
        return None, None
    return box, raw_mask


def _median_gray(pairs):
    grays = [cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) for a, _ in pairs]
    return np.median(np.stack(grays, axis=0), axis=0).astype(np.float32)


def _find_edge_outward(a, band_lo, band_hi, start, max_grow, fallback=None, motion=None):
    fallback = start if fallback is None else fallback
    lo = max(1, start - max_grow)
    flat_n = CROP_EDGE_FLAT_ROWS
    band = a[:, band_lo:band_hi]
    mband = motion[:, band_lo:band_hi] if motion is not None else None
    row_std_all = band.std(axis=1)
    min_run = int(a.shape[0] * CROP_EDGE_MIN_FLAT_RUN)
    for y in range(start, lo - 1, -1):
        inner = band[y:y + 2].mean(axis=0)
        outer = band[max(0, y - 2):y].mean(axis=0) if y >= 1 else None
        if outer is None:
            return fallback
        beyond = band[max(0, y - flat_n):y]
        if beyond.shape[0] < flat_n and y - flat_n > 0:
            continue
        if beyond.shape[0] >= 3:
            row_std = beyond.std(axis=1)
            row_mean = beyond.mean(axis=1)
            if row_std.max() > CROP_EDGE_FLAT_STD or (row_mean.max() - row_mean.min()) > CROP_EDGE_FLAT_STD:
                continue

        d = outer - inner
        med = float(np.median(d))
        step_edge = False
        if abs(med) >= CROP_EDGE_STEP_MIN:
            consistent = np.mean((np.sign(d) == np.sign(med)) & (np.abs(d) >= CROP_EDGE_STEP_MIN / 2))
            step_edge = consistent >= CROP_EDGE_COL_CONSISTENCY

        texture_edge = (
            float(band[y:y + 4].std(axis=1).mean()) >= CROP_EDGE_TEXTURE_STD
            and mband is not None
            and float(mband[y:y + 4].mean()) >= CROP_EDGE_MOTION_AT_EDGE
        )

        if step_edge or texture_edge:
            k, run = y - 1, 0
            while k >= 0 and row_std_all[k] < CROP_EDGE_FLAT_STD:
                run += 1
                k -= 1
            if k < 0 or run >= min_run:
                return y
    return fallback


def refine_box_to_edges(median_gray, seed, motion_mask=None):
    H, W = median_gray.shape
    x, y, w, h = seed
    x1, y1, x2, y2 = x, y, x + w, y + h
    gx = int(W * CROP_EDGE_MAX_GROW_X)
    gy = int(H * CROP_EDGE_MAX_GROW_Y)

    motion = None
    if motion_mask is not None:
        motion = cv2.resize(motion_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    def inset(size):
        return max(8, int(size * 0.04))

    g, mo = median_gray, motion
    for _ in range(2):
        ix, iy = inset(x2 - x1), inset(y2 - y1)
        y1 = _find_edge_outward(g, x1, x2, y1 + iy, gy + iy, fallback=y1, motion=mo)
        y2 = H - _find_edge_outward(g[::-1], x1, x2, H - y2 + iy, gy + iy, fallback=H - y2,
                                    motion=None if mo is None else mo[::-1])
        x1 = _find_edge_outward(g.T, y1, y2, x1 + ix, gx + ix, fallback=x1,
                                motion=None if mo is None else mo.T)
        x2 = W - _find_edge_outward(g.T[::-1], y1, y2, W - x2 + ix, gx + ix, fallback=W - x2,
                                    motion=None if mo is None else mo.T[::-1])

    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def compute_safe_crop(clip):
    frame_w, frame_h = clip.w, clip.h
    box = None
    try:
        pairs = _sample_frame_pairs(clip)
        seed, raw_mask = find_motion_seed(pairs, frame_w, frame_h)
        if seed is not None:
            refined = refine_box_to_edges(_median_gray(pairs), seed, raw_mask)
            if _validate_box(refined, frame_w, frame_h):
                box = refined
            elif _validate_box(seed, frame_w, frame_h):
                box = seed
    except Exception as e:
        print(f"⚠️ Detecció per moviment ha fallat: {e}")

    if box is None:
        return None

    x, y, w, h = box
    mx, my = int(w * CROP_INNER_MARGIN), int(h * CROP_INNER_MARGIN)
    x1, y1 = max(0, x + mx), max(0, y + my)
    x2, y2 = min(frame_w, x + w - mx), min(frame_h, y + h - my)
    x1, y1 = x1 - (x1 % 2), y1 - (y1 % 2)
    x2, y2 = x2 - (x2 % 2), y2 - (y2 % 2)
    return (x1, y1, x2, y2)


# ==========================================
# MINIATURA EDITORIAL
# ==========================================

def create_editorial_thumbnail(video_path, thumbnail_title, output_path=os.path.join(VIDEOS_DIR, "final_thumbnail.jpg")):
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    clip = VideoFileClip(video_path)
    crop_box = compute_safe_crop(clip)

    t_sample = min(1.0, max(clip.duration - 0.1, 0.5)) if clip.duration else 0.5
    frame_np = clip.get_frame(t_sample)
    clip.close()

    frame_pil = Image.fromarray(frame_np)
    cropped_frame = frame_pil.crop(crop_box) if crop_box else frame_pil

    canvas = Image.new("RGBA", (1080, 1920), (0, 0, 0, 255))

    w_c, h_c = cropped_frame.size
    scale_fg = 1080 / w_c
    fg_w = 1080
    fg_h = int(h_c * scale_fg)

    fg_resized = cropped_frame.resize((fg_w, fg_h), Image.Resampling.LANCZOS)
    fg_blurred = fg_resized.filter(ImageFilter.GaussianBlur(radius=32)).convert("RGBA")
    dark_tint = Image.new("RGBA", (fg_w, fg_h), (0, 0, 0, 130))
    fg_box = Image.alpha_composite(fg_blurred, dark_tint)

    fg_y = max(0, (1920 - fg_h) // 2)
    canvas.paste(fg_box, (0, fg_y), fg_box)

    draw = ImageDraw.Draw(canvas)
    title_font = get_jakarta_font("bold", size=76)
    words = thumbnail_title.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        w = draw.textbbox((0, 0), test_line, font=title_font)[2]
        if w <= 920:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    if current_line:
        lines.append(" ".join(current_line))

    line_h = 96
    total_title_h = len(lines) * line_h
    logo_size = 190
    gap = 46
    total_block_h = total_title_h + gap + logo_size
    start_y = 960 - (total_block_h // 2)

    text_y = start_y
    for line in lines:
        w = draw.textbbox((0, 0), line, font=title_font)[2]
        x = (1080 - w) // 2
        draw.text((x + 4, text_y + 4), line, font=title_font, fill=(0, 0, 0, 240))
        draw.text((x, text_y), line, font=title_font, fill=COLOR_WHITE)
        text_y += line_h

    logo_x = (1080 - logo_size) // 2
    logo_y = start_y + total_title_h + gap

    logo_file = LOGO_PATH if os.path.exists(LOGO_PATH) else ("logo.png" if os.path.exists("logo.png") else None)
    if logo_file:
        try:
            logo_img = Image.open(logo_file).convert("RGBA").resize((logo_size, logo_size), Image.Resampling.LANCZOS)
            mask = Image.new("L", (logo_size, logo_size), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, logo_size, logo_size), fill=255)
            canvas.paste(logo_img, (logo_x, logo_y), mask)
        except Exception:
            pass
    else:
        draw.ellipse([logo_x, logo_y, logo_x + logo_size, logo_y + logo_size], fill=COLOR_YELLOW)
        f_font = get_jakarta_font("bold", size=int(logo_size * 0.65))
        f_bbox = draw.textbbox((0, 0), "f", font=f_font)
        f_w = f_bbox[2] - f_bbox[0]
        f_h = f_bbox[3] - f_bbox[1]
        draw.text((logo_x + (logo_size - f_w) // 2 - f_bbox[0], logo_y + (logo_size - f_h) // 2 - f_bbox[1]), "f", font=f_font, fill=COLOR_WHITE)

    bg_rgb = canvas.convert("RGB")
    bg_rgb.save(output_path, "JPEG", quality=95)
    print(f"🖼️ Miniatura editorial generada amb èxit a: {output_path}")
    return np.array(bg_rgb), output_path


# ==========================================
# RENDERITZAT DE DOS VÍDEOS EN PARAL·LEL
# ==========================================

def process_dual_video_canvases(input_path, fb_text, short_hook, thumbnail_img_np, output_fb_path, output_shorts_path):
    """
    Optimitza el renderitzat processant el crop del vídeo un sol cop i
    exportant les dues variants: Facebook (llarg) i Shorts (curt en gran).
    """
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    clip = VideoFileClip(input_path)
    crop_box = compute_safe_crop(clip)

    if crop_box:
        x1, y1, x2, y2 = crop_box
        cropped_clip = clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
    else:
        cropped_clip = clip

    scaled_clip = cropped_clip.resized(width=1080)
    cover_clip = ImageClip(thumbnail_img_np).with_duration(1.0 / 30.0)

    # 1. GENERAR VÍDEO FACEBOOK (Capçalera Tweet clàssica)
    print("🎨 [1/2] Renderitzant composició per a Facebook...")
    header_fb_np = create_facebook_header_image(fb_text, width=1080)
    header_fb_h = header_fb_np.shape[0]

    header_fb_clip = ImageClip(header_fb_np).with_duration(scaled_clip.duration).with_position(("center", 180))
    video_fb_pos = scaled_clip.with_position(("center", 180 + header_fb_h + 10))
    composite_fb = CompositeVideoClip([video_fb_pos, header_fb_clip], size=(1080, 1920))
    final_fb_video = concatenate_videoclips([cover_clip, composite_fb])

    final_fb_video.write_videofile(
        output_fb_path,
        codec="libx264",
        audio_codec="aac",
        fps=30,
        preset="fast"
    )

    header_fb_clip.close()
    composite_fb.close()
    final_fb_video.close()

    # 2. GENERAR VÍDEO SHORTS (Instagram / TikTok / Snapchat)
    print("🎨 [2/2] Renderitzant composició per a Shorts (IG/TikTok/Snapchat)...")
    header_shorts_np = create_shorts_header_image(short_hook, width=1080)
    header_shorts_h = header_shorts_np.shape[0]

    header_shorts_clip = ImageClip(header_shorts_np).with_duration(scaled_clip.duration).with_position(("center", 140))
    video_shorts_pos = scaled_clip.with_position(("center", 140 + header_shorts_h + 15))
    composite_shorts = CompositeVideoClip([video_shorts_pos, header_shorts_clip], size=(1080, 1920))
    final_shorts_video = concatenate_videoclips([cover_clip, composite_shorts])

    final_shorts_video.write_videofile(
        output_shorts_path,
        codec="libx264",
        audio_codec="aac",
        fps=30,
        preset="fast"
    )

    header_shorts_clip.close()
    composite_shorts.close()
    final_shorts_video.close()

    cover_clip.close()
    scaled_clip.close()
    cropped_clip.close()
    clip.close()


# ==========================================
# NOTIFICACIÓ TELEGRAM
# ==========================================

def send_telegram_notification(video_fb_path, video_shorts_path, thumbnail_path, fb_text, short_hook, credits, generated_caption, video_id, tracking_id):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Notificació de Telegram omessa (tokens no configurats).")
        return

    safe_fb_text = html.escape(fb_text or "")
    safe_short_hook = html.escape(short_hook or "")
    safe_credits = html.escape(credits or "No especificada")
    safe_tracking_id = html.escape(tracking_id or video_id or "")

    if TEST_MODE:
        print("🧪 [Mode Proves] Enviant tots dos vídeos, portada i caption per Telegram...")

        # Vídeo 1: Versió Facebook
        caption_v1 = (
            f"🎬 <b>[TEST MODE] 1/2 VERSIÓ FACEBOOK (Text Llarg)</b>\n\n"
            f"📌 <b>Text</b>:\n<i>{safe_fb_text}</i>\n\n"
            f"👤 <b>Font Original</b>: {safe_credits}\n"
            f"🆔 <b>ID</b>: <code>{safe_tracking_id}</code>"
        )
        url_video = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
        with open(video_fb_path, "rb") as vf1:
            requests.post(url_video, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption_v1, "parse_mode": "HTML"}, files={"video": vf1})

        # Vídeo 2: Versió Shorts (IG / TikTok / Snapchat)
        caption_v2 = (
            f"🎬 <b>[TEST MODE] 2/2 VERSIÓ SHORTS (IG / TikTok / Snapchat)</b>\n\n"
            f"📌 <b>Hook Gran</b>:\n<i>{safe_short_hook}</i>\n\n"
            f"🆔 <b>ID</b>: <code>{safe_tracking_id}</code>"
        )
        with open(video_shorts_path, "rb") as vf2:
            requests.post(url_video, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption_v2, "parse_mode": "HTML"}, files={"video": vf2})

        # Portada
        if thumbnail_path and os.path.exists(thumbnail_path):
            url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(thumbnail_path, "rb") as photo_file:
                requests.post(url_photo, data={"chat_id": TELEGRAM_CHAT_ID, "caption": "🖼️ <b>[TEST MODE] Portada generada</b>", "parse_mode": "HTML"}, files={"photo": photo_file})

        # Caption complet amb l'ID
        url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        caption_text = f"📝 <b>[TEST MODE] CAPTION AMB ID</b>:\n\n<code>{html.escape(generated_caption)}</code>"
        requests.post(url_msg, data={"chat_id": TELEGRAM_CHAT_ID, "text": caption_text, "parse_mode": "HTML"})

    else:
        print("🚀 [Producció] Enviant només resum de confirmació...")
        url_msg = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        confirm_text = (
            f"✅ <b>VÍDEOS PUBLICATS A XARXES (DUAL PIPELINE)</b>\n\n"
            f"🆔 <b>ID:</b> <code>{safe_tracking_id}</code>\n"
            f"📌 <b>Hook Shorts:</b> {safe_short_hook}\n"
            f"📘 <b>Facebook:</b> Versió Text Llarg enviada\n"
            f"📱 <b>IG / TikTok / Snap:</b> Versió Shorts enviada\n"
            f"👤 <b>Font:</b> {safe_credits}\n"
            f"🌐 <i>Estat a sources.csv: <b>done</b></i>"
        )
        requests.post(url_msg, data={"chat_id": TELEGRAM_CHAT_ID, "text": confirm_text, "parse_mode": "HTML"})


# ==========================================
# DESCARREGA I PREPARACIÓ
# ==========================================

def _cleanup_temp_input():
    for f in glob.glob("temp_input.*"):
        try:
            os.remove(f)
        except OSError:
            pass


def get_reel_by_url(reel_url):
    shortcode = extract_shortcode(reel_url)
    tracking_id = shortcode or str(int(time.time()))

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
    except Exception as e:
        print(f"❌ Error descarregant reel amb yt-dlp: {e}")
        return None

    video_id = str(info.get("id") or shortcode or reel_url)
    downloaded_path = ydl.prepare_filename(info)
    if not os.path.exists(downloaded_path):
        candidates = glob.glob("temp_input.*")
        downloaded_path = candidates[0] if candidates else None

    if not downloaded_path or not os.path.exists(downloaded_path):
        return None

    caption_raw = info.get("description") or ""

    # Dades per al tracking CSV
    uploader = info.get("uploader") or info.get("channel") or info.get("uploader_id") or ""
    source_account = f"@{uploader}" if uploader else "Desconegut"
    initial_likes = get_reel_likes_count(reel_url) or info.get("like_count") or 0

    frame_image = extract_frame_as_image(downloaded_path, timestamp=0.5)
    ai_result = analyze_content_with_retry(
        frame_image,
        caption_raw=caption_raw,
        reel_url=reel_url,
        tracking_id=tracking_id,
        max_retries=5,
        delay_seconds=60
    )

    if not ai_result or len(ai_result) != 5:
        return None

    credits, fb_text, short_hook, generated_caption, thumbnail_title = ai_result

    return {
        "downloaded_path": downloaded_path,
        "video_id": video_id,
        "tracking_id": tracking_id,
        "shortcode": shortcode,
        "source_account": source_account,
        "initial_likes": initial_likes,
        "credits": credits,
        "fb_text": fb_text,
        "short_hook": short_hook,
        "generated_caption": generated_caption,
        "thumbnail_title": thumbnail_title
    }


# ==========================================
# FLUX PRINCIPAL
# ==========================================

def main():
    if not os.path.exists("sources.csv"):
        print("❌ No s'ha trobat el fitxer sources.csv")
        return

    os.makedirs(VIDEOS_DIR, exist_ok=True)

    pending_urls = []
    with open("sources.csv", mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 1:
                continue
            url = row[0].strip()
            status = row[1].strip().lower() if len(row) > 1 else "pending"

            if (status == "pending" or TEST_MODE) and url.startswith("http"):
                pending_urls.append(url)

    if not pending_urls:
        print("ℹ️ No hi ha cap reel amb estat 'pending' a sources.csv.")
        return

    for reel_url in pending_urls:
        print(f"\n🚀 Processant reel: {reel_url}")
        reel_data = get_reel_by_url(reel_url)

        if not reel_data or not reel_data.get("downloaded_path"):
            print(f"⚠️ Reel omès o fallat: {reel_url}")
            update_csv_status(reel_url, "failed")
            continue

        video_file = reel_data["downloaded_path"]
        video_id = reel_data["video_id"]
        tracking_id = reel_data["tracking_id"]
        shortcode = reel_data["shortcode"]
        source_account = reel_data["source_account"]
        initial_likes = reel_data["initial_likes"]
        credits = reel_data["credits"]
        fb_text = reel_data["fb_text"]
        short_hook = reel_data["short_hook"]
        generated_caption = reel_data["generated_caption"]
        thumbnail_title = reel_data["thumbnail_title"]

        # Rutes dels dos fitxers de vídeo
        unique_video_fb_rel = f"video_{video_id}_fb.mp4"
        unique_video_fb_path = os.path.join(VIDEOS_DIR, unique_video_fb_rel)

        unique_video_shorts_rel = f"video_{video_id}_shorts.mp4"
        unique_video_shorts_path = os.path.join(VIDEOS_DIR, unique_video_shorts_rel)

        thumbnail_rel = "final_thumbnail.jpg"
        thumbnail_path = os.path.join(VIDEOS_DIR, thumbnail_rel)

        # 1. Generar la miniatura editorial
        thumbnail_np, thumbnail_file = create_editorial_thumbnail(video_file, thumbnail_title, thumbnail_path)

        # 2. Generar els dos vídeos (.mp4 de Facebook i .mp4 de Shorts)
        process_dual_video_canvases(
            video_file,
            fb_text,
            short_hook,
            thumbnail_np,
            unique_video_fb_path,
            unique_video_shorts_path
        )

        # 3. Guardar IDs i estats en local ABANS de fer el push a GitHub
        save_processed_id(video_id, shortcode=shortcode)
        update_csv_status(reel_url, "done")

        # 4. Sincronitzar amb GitHub
        push_media_to_github([unique_video_fb_rel, unique_video_shorts_rel, thumbnail_rel])

        snapchat_ok = False
        # 5. Publicació multicanal
        if not TEST_MODE:
            # Publicació a Buffer (Facebook rep video_fb, Instagram i TikTok reben video_shorts)
            publish_to_buffer(
                generated_caption,
                fb_video_filename=unique_video_fb_rel,
                shorts_video_filename=unique_video_shorts_rel,
                thumbnail_offset_ms=0
            )

            # Publicació selectiva a Snapchat Spotlight (TOP 20 del mes) amb video_shorts
            if should_publish_to_snapchat(reel_url, unique_video_shorts_path):
                snapchat_ok = publish_to_bundle_snapchat(
                    unique_video_shorts_path,
                    thumbnail_title,
                    short_hook,
                    tracking_id
                )
                if snapchat_ok:
                    record_snapchat_publication()

            # Guardar el registre al CSV de tracking
            record_published_post_tracking(
                tracking_id=tracking_id,
                shortcode=shortcode,
                source_url=reel_url,
                source_account=source_account,
                initial_likes=initial_likes,
                thumbnail_title=thumbnail_title,
                snapchat_published=snapchat_ok
            )
        else:
            print("🧪 [Mode Proves Actiu]: S'omet la publicació externa i el tracking CSV.")

        # 6. Notificació per Telegram segons el mode
        send_telegram_notification(
            unique_video_fb_path,
            unique_video_shorts_path,
            thumbnail_file,
            fb_text,
            short_hook,
            credits,
            generated_caption,
            video_id,
            tracking_id
        )

        print("✅ Procés finalitzat amb èxit!")
        break


if __name__ == "__main__":
    main()
