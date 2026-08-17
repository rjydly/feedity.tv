import os
import re
import json
import numpy as np
import cv2
import ollama
from moviepy import VideoFileClip, TextClip, CompositeVideoClip

def _sample_frames_grayscale_from_clip(clip, num_samples=10):
    """Extreu fotogrames distribuïts uniformement en escala de grises."""
    duration = clip.duration
    if duration <= 0:
        return []
    
    times = np.linspace(0, duration, num=num_samples, endpoint=False)
    frames = []
    for t in times:
        frame = clip.get_frame(t)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frames.append(gray)
    return frames

def _refine_micro_borders(frame_crop, threshold_white=215, max_check_pixels=15):
    """
    FASE 2: Micro-crop de precisió.
    Inspecciona els 4 extrems de la imatge ja retallada per eliminar
    petites línies o franjes blanques/clares residuals.
    """
    h, w = frame_crop.shape
    top, bottom, left, right = 0, 0, 0, 0

    # 1. Part superior (Top)
    for y in range(min(max_check_pixels, h)):
        if np.mean(frame_crop[y, :]) > threshold_white:
            top = y + 1
        else:
            break

    # 2. Part inferior (Bottom)
    for y in range(1, min(max_check_pixels, h)):
        if np.mean(frame_crop[h - y, :]) > threshold_white:
            bottom = y
        else:
            break

    # 3. Extrem esquerre (Left)
    for x in range(min(max_check_pixels, w)):
        if np.mean(frame_crop[:, x]) > threshold_white:
            left = x + 1
        else:
            break

    # 4. Extrem dret (Right)
    for x in range(1, min(max_check_pixels, w)):
        if np.mean(frame_crop[:, w - x]) > threshold_white:
            right = x
        else:
            break

    return top, bottom, left, right

def crop_content_bounding_box(clip):
    """
    Detecció i retall de fons en 2 FASES:
    - Fase 1: Crop Global (Gradient Sobel + Variància de moviment).
    - Fase 2: Micro-Crop de precisió sobre franjes perifèriques clares.
    """
    frames = _sample_frames_grayscale_from_clip(clip)
    if not frames:
        return None

    stacked = np.stack(frames, axis=0)
    mean_frame = stacked.mean(axis=0).astype(np.uint8)

    # =========================================================================
    # FASE 1: Crop Global
    # =========================================================================
    grad_x = cv2.Sobel(mean_frame, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mean_frame, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    _, edge_mask = cv2.threshold(magnitude.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)

    variance_map = stacked.std(axis=0)
    _, motion_mask = cv2.threshold(variance_map.astype(np.uint8), 3, 255, cv2.THRESH_BINARY)

    combined_mask = cv2.bitwise_or(edge_mask, motion_mask)
    kernel = np.ones((9, 9), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h_f, w_f = mean_frame.shape
    valid_contours = [c for c in contours if cv2.boundingRect(c)[2] > 100 and cv2.boundingRect(c)[3] > 100]

    if not valid_contours:
        return None

    x_min, y_min = w_f, h_f
    x_max, y_max = 0, 0

    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    w_phase1 = x_max - x_min
    h_phase1 = y_max - y_min

    # =========================================================================
    # FASE 2: Micro-Crop de Precisió
    # =========================================================================
    cropped_subframe = mean_frame[y_min:y_max, x_min:x_max]
    top_trim, bottom_trim, left_trim, right_trim = _refine_micro_borders(cropped_subframe)

    final_x = x_min + left_trim
    final_y = y_min + top_trim
    final_w = w_phase1 - left_trim - right_trim
    final_h = h_phase1 - top_trim - bottom_trim

    if top_trim or bottom_trim or left_trim or right_trim:
        print(f"🔍 Micro-crop Fase 2 aplicat: Top={top_trim}px, Bottom={bottom_trim}px, Left={left_trim}px, Right={right_trim}px")

    return (final_x, final_y, final_w, final_h)

def analyze_caption_with_local_ai(caption):
    """
    Processa el text amb Gemma 2 forçant sortida en Anglès,
    titulars obligatoris en negreta i neteja de CTAs de tercers.
    """
    prompt = f"""
    You are the head social media copywriter for @feedity.tv.
    Analyze the following raw video caption and generate the requested fields.

    Raw caption: "{caption}"

    STRICT INSTRUCTIONS:
    1. **LANGUAGE**: ALL generated content MUST be written strictly in ENGLISH. Do not use Spanish, Catalan, or any other language.
    2. **credits**: Extract the original creator's social handle (e.g., @creator). If not explicitly mentioned, return "Unknown".
    3. **headline**: Create ONE SHORT, IMPACTFUL HEADLINE in ENGLISH (max 6-8 words). Use <b> and </b> tags around the main keywords to make them BOLD. NEVER leave this blank, NEVER return "Feedity Media", and NEVER leave it empty.
    4. **generated_caption**: Write an engaging, viral Instagram caption in ENGLISH.
       - For any Call to Action (CTA) asking users to follow, ONLY use @feedity.tv (NEVER mention third-party accounts like @FBOY or others).
       - Keep it snappy and engaging, using relevant emojis and trending English hashtags. Do NOT write long Wikipedia-style summaries.

    Respond ONLY with a valid JSON object matching this schema:
    {{
      "credits": "@username",
      "headline": "This is a <b>viral headline</b>",
      "generated_caption": "Follow @feedity.tv for more viral clips! 🍿..."
    }}
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
            credits = data.get("credits", "Unknown")
            headline = data.get("headline", "").strip()
            gen_caption = data.get("generated_caption", "").strip()
            
            # Fallback en anglès si no hi ha headline vàlid
            if not headline or headline.lower() in ["feedity media", "<b>feedity</b> media"]:
                headline = "<b>Viral</b> moment"

            # Sanitització de seguretat per forçar la nostra marca
            gen_caption = re.sub(r'@[A-Za-z0-9_.]+', '@feedity.tv', gen_caption)
            
            if credits and credits != "Unknown":
                gen_caption = f"{gen_caption}\n\nVia: {credits}"
                
            return credits, headline, gen_caption
    except Exception as e:
        print(f"⚠️ Error en analitzar amb Ollama: {e}")
    
    return "Unknown", "<b>Featured</b> clip", f"Follow @feedity.tv for more! 🍿\n\n{caption}"
