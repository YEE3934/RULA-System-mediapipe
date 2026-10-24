import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import math
import numpy as np
import pandas as pd
import os
import urllib.request
import zipfile
import io
import gc
import logging
import base64
import tempfile
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Streamlit Cloud: keep MediaPipe legacy Pose at complexity 1.
# Complexity 2 asks MediaPipe to download pose_landmark_heavy.tflite into site-packages,
# which is typically read-only on hosted deployments.
POSE_MODEL_COMPLEXITY = 1
APP_BUILD_TAG = "RULA_CLOUD_POSE_FIX_2026-10-24_v2"

# ==================== 【正統 RULA 查表矩陣】 ====================
TABLE_A = {
    1: { 1: {1: {1:1, 2:2}, 2: {1:2, 2:2}, 3: {1:2, 2:3}, 4: {1:3, 2:3}},
         2: {1: {1:2, 2:2}, 2: {1:2, 2:3}, 3: {1:3, 2:3}, 4: {1:3, 2:4}},
         3: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}} },
    2: { 1: {1: {1:1, 2:2}, 2: {1:2, 2:2}, 3: {1:3, 2:3}, 4: {1:3, 2:4}},
         2: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}},
         3: {1: {1:3, 2:4}, 2: {1:3, 2:4}, 3: {1:4, 2:4}, 4: {1:4, 2:5}} },
    3: { 1: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:3, 2:4}, 4: {1:4, 2:4}},
         2: {1: {1:3, 2:3}, 2: {1:3, 2:4}, 3: {1:4, 2:4}, 4: {1:4, 2:5}},
         3: {1: {1:3, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:5}} },
    4: { 1: {1: {1:3, 2:4}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:5, 2:5}},
         2: {1: {1:4, 2:4}, 2: {1:4, 2:5}, 3: {1:4, 2:5}, 4: {1:5, 2:6}},
         3: {1: {1:4, 2:5}, 2: {1:5, 2:5}, 3: {1:5, 2:6}, 4: {1:6, 2:7}} },
    5: { 1: {1: {1:4, 2:4}, 2: {1:4, 2:5}, 3: {1:4, 2:5}, 4: {1:5, 2:6}},
         2: {1: {1:4, 2:5}, 2: {1:5, 2:5}, 3: {1:5, 2:6}, 4: {1:6, 2:7}},
         3: {1: {1:5, 2:6}, 2: {1:6, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}} },
    6: { 1: {1: {1:5, 2:5}, 2: {1:5, 2:6}, 3: {1:5, 2:6}, 4: {1:6, 2:7}},
         2: {1: {1:5, 2:6}, 2: {1:6, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}},
         3: {1: {1:6, 2:7}, 2: {1:7, 2:7}, 3: {1:7, 2:8}, 4: {1:8, 2:8}} }
}

TABLE_B = {
    1: {1: {1:1, 2:3}, 2: {1:2, 2:3}, 3: {1:3, 2:4}, 4: {1:5, 2:5}, 5: {1:6, 2:6}, 6: {1:7, 2:7}},
    2: {1: {1:2, 2:3}, 2: {1:3, 2:3}, 3: {1:4, 2:4}, 4: {1:5, 2:5}, 5: {1:6, 2:7}, 6: {1:7, 2:8}},
    3: {1: {1:3, 2:3}, 2: {1:4, 2:4}, 3: {1:4, 2:5}, 4: {1:6, 2:6}, 5: {1:7, 2:7}, 6: {1:7, 2:8}},
    4: {1: {1:5, 2:5}, 2: {1:5, 2:6}, 3: {1:6, 2:7}, 4: {1:7, 2:7}, 5: {1:8, 2:8}, 6: {1:8, 2:8}},
    5: {1: {1:6, 2:6}, 2: {1:6, 2:7}, 3: {1:7, 2:8}, 4: {1:8, 2:8}, 5: {1:8, 2:8}, 6: {1:8, 2:8}},
    6: {1: {1:7, 2:7}, 2: {1:7, 2:8}, 3: {1:8, 2:8}, 4: {1:8, 2:8}, 5: {1:8, 2:8}, 6: {1:8, 2:8}}
}

TABLE_C = [
    [1, 2, 3, 3, 4, 5, 5], [2, 2, 3, 4, 4, 5, 5], [3, 3, 3, 4, 4, 5, 6], [3, 3, 3, 4, 5, 6, 6],
    [4, 4, 4, 5, 6, 7, 7], [4, 4, 5, 6, 6, 7, 7], [5, 5, 6, 6, 7, 7, 7], [5, 5, 6, 7, 7, 7, 7]
]

# ==================== 【自動中文字型載入器】 ====================
@st.cache_resource
def ensure_chinese_font():
    candidates = [
        "font.ttf",
        "/tmp/chinese_font.otf",
        "/tmp/chinese_font.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msjh.ttc",
        "msjh.ttc"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    if os.path.exists("/usr/share/fonts"):
        for root, _, files in os.walk("/usr/share/fonts"):
            for f in files:
                if any(k in f.lower() for k in ["wqy", "noto", "cjk", "zenhei", "microhei", "droid"]):
                    return os.path.join(root, f)

    urls = [
        ("https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf", "/tmp/chinese_font.otf"),
        ("https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf", "/tmp/chinese_font.ttf")
    ]
    for url, target_path in urls:
        try:
            urllib.request.urlretrieve(url, target_path)
            if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
                return target_path
        except Exception:
            continue
    return None

FONT_FILE_PATH = ensure_chinese_font()

def put_chinese_text(img, text, position, color, size=18):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = None
    if FONT_FILE_PATH and os.path.exists(FONT_FILE_PATH):
        try:
            font = ImageFont.truetype(FONT_FILE_PATH, size)
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()
    
    x, y = position
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 1), (1, 0)]:
        draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(color[2], color[1], color[0]), font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==================== 【繞過 /media/ 的瀏覽器直出下載模組】 ====================
def render_blob_download_button(zip_bytes, filename, label_text):
    b64_zip = base64.b64encode(zip_bytes).decode("utf-8")
    html_code = f"""
    <div style="margin: 10px 0;">
        <button id="dl_btn" style="
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            padding: 14px 20px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        ">{label_text}</button>
    </div>
    <script>
    document.getElementById("dl_btn").addEventListener("click", function() {{
        const b64Data = "{b64_zip}";
        const byteCharacters = atob(b64Data);
        const byteArrays = [];
        const sliceSize = 1024;
        for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {{
            const slice = byteCharacters.slice(offset, offset + sliceSize);
            const byteNumbers = new Array(slice.length);
            for (let i = 0; i < slice.length; i++) {{
                byteNumbers[i] = slice.charCodeAt(i);
            }}
            const byteArray = new Uint8Array(byteNumbers);
            byteArrays.push(byteArray);
        }}
        const blob = new Blob(byteArrays, {{type: "application/zip"}});
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        a.download = "{filename}";
        document.body.appendChild(a);
        a.click();
        setTimeout(function() {{
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }}, 2000);
    }});
    </script>
    """
    components.html(html_code, height=70)

# ==================== 【幾何與人因工程計分邏輯】 ====================
def calculate_2d_angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-5 or norm_bc < 1e-5:
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def calculate_3d_angle(a, b, c):
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-5 or norm_bc < 1e-5:
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def enhance_image_for_pose(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ==================== 【姿態偵測重試與關鍵點品質檢查】 ====================
# Visibility 是模型對關鍵點可見性的估計，不是臨床級準確率。
POSE_MIN_VISIBILITY = 0.25
POSE_RETRY_MEAN_VISIBILITY = 0.58


def _visibility_values(results, indices):
    if not results or not getattr(results, "pose_landmarks", None):
        return []
    landmarks = results.pose_landmarks.landmark
    values = []
    for idx in indices:
        if idx >= len(landmarks):
            values.append(0.0)
            continue
        value = float(getattr(landmarks[idx], "visibility", 0.0) or 0.0)
        values.append(value if np.isfinite(value) else 0.0)
    return values


def detect_pose_with_fallback(bgr_img, pose_model, required_indices=None):
    """先用原圖；品質不足時再試 CLAHE 與放大圖，回傳品質最佳的結果。"""
    if bgr_img is None or bgr_img.size == 0:
        return None, 0.0, "empty_frame"

    if required_indices is None:
        required_indices = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    h, w = bgr_img.shape[:2]
    attempts = [(bgr_img, "original")]
    if h >= 2 and w >= 2:
        try:
            attempts.append((enhance_image_for_pose(bgr_img), "clahe"))
        except cv2.error:
            pass
        # 小人像／遠距離影片常因像素太少而失準；只在必要時才跑放大版本。
        up_scale = min(2.0, 1600.0 / max(h, w))
        up_w, up_h = max(1, int(w * up_scale)), max(1, int(h * up_scale))
        attempts.append((cv2.resize(bgr_img, (up_w, up_h), interpolation=cv2.INTER_CUBIC), "upscale"))

    best_results, best_score, best_method = None, -1.0, "none"
    for attempt_img, method in attempts:
        try:
            rgb = cv2.cvtColor(attempt_img, cv2.COLOR_BGR2RGB)
            results = pose_model.process(rgb)
        except Exception:
            logging.exception("MediaPipe pose detection attempt failed (%s)", method)
            continue

        if not results or not getattr(results, "pose_landmarks", None):
            continue
        values = _visibility_values(results, required_indices)
        if not values:
            continue
        mean_vis = float(np.mean(values))
        min_vis = float(np.min(values))
        # 同時考量平均與低可見度點，避免一兩個很差的關鍵點被平均值掩蓋。
        quality = 0.65 * mean_vis + 0.35 * min_vis
        if quality > best_score:
            best_results, best_score, best_method = results, quality, method

        if mean_vis >= POSE_RETRY_MEAN_VISIBILITY and min_vis >= POSE_MIN_VISIBILITY:
            break

    if best_results is None:
        return None, 0.0, best_method
    return best_results, max(0.0, best_score), best_method


def landmarks_meet_visibility(results, required_indices, threshold=POSE_MIN_VISIBILITY):
    values = _visibility_values(results, required_indices)
    return bool(values) and all(v >= threshold for v in values)

def get_upper_arm_score(angle, is_abducted=False, is_raised=False, is_supported=False, is_extended=False):
    """RULA upper-arm base score + modifiers; extension direction needs visual/manual confirmation."""
    if is_extended and angle > 20:
        base = 2  # >20° extension: score 2 (do not treat it as forward flexion)
    elif angle <= 20:
        base = 1
    elif angle <= 45:
        base = 2
    elif angle <= 90:
        base = 3
    else:
        base = 4
    if is_abducted:
        base += 1
    if is_raised:
        base += 1
    if is_supported:
        base -= 1
    return min(max(base, 1), 6)


def get_lower_arm_score(angle, is_across_midline=False, is_out_to_side=False):
    base = 1 if 60 <= angle <= 100 else 2
    # RULA adds one point if working across the midline OR out to the side.
    if is_across_midline or is_out_to_side:
        base += 1
    return min(max(base, 1), 3)


def get_wrist_score(angle, is_deviated=False):
    if angle <= 5:
        base = 1
    elif angle <= 15:
        base = 2
    else:
        base = 3
    if is_deviated:
        base += 1
    return min(max(base, 1), 4)


def get_neck_score(angle, is_twisted=False, is_side_bending=False, is_extended=False):
    if is_extended:
        base = 4  # Neck extension is a distinct RULA category; unsigned 2D angles cannot infer it reliably.
    elif angle <= 10:
        base = 1
    elif angle <= 20:
        base = 2
    else:
        base = 3
    if is_twisted:
        base += 1
    if is_side_bending:
        base += 1
    return min(max(base, 1), 6)


def get_trunk_score(angle, is_twisted=False, is_side_bending=False, is_extended=False):
    if is_extended:
        base = 2
    elif angle <= 5:
        base = 1
    elif angle <= 20:
        base = 2
    elif angle <= 60:
        base = 3
    else:
        base = 4
    if is_twisted:
        base += 1
    if is_side_bending:
        base += 1
    return min(max(base, 1), 6)

def compute_rula_full(ua, la, w, wt, nk, tk, lg, m_val, f_val):
    ua = min(max(ua, 1), 6)
    la = min(max(la, 1), 3)
    w  = min(max(w, 1), 4)
    wt = min(max(wt, 1), 2)
    nk = min(max(nk, 1), 6)
    tk = min(max(tk, 1), 6)
    lg = min(max(lg, 1), 2)
    score_a = TABLE_A[ua][la][w][wt]
    score_b = TABLE_B[nk][tk][lg]
    score_c = score_a + m_val + f_val
    score_d = score_b + m_val + f_val
    grand_score = TABLE_C[min(score_c, 8) - 1][min(score_d, 7) - 1]
    al_num = 1 if grand_score <= 2 else (2 if grand_score <= 4 else (3 if grand_score <= 6 else 4))
    return score_a, score_b, score_c, score_d, grand_score, f"AL{al_num}"

# ==================== 【全 33 點生物力學渲染器 (全身雙側完整繪製 + 目標側高亮標註)】 ====================
def draw_full_biomechanical_skeleton(
    img, pts_dict, ang_ua, ang_la, ang_wr, ang_tk, ang_nk, wt_score, target_side_label=""
):
    """
    完整繪製全身 33 個解剖點：
    1. 臉部五官 11 點網格 (鼻、雙眼內中外、雙耳、嘴角)
    2. 全身左右雙側骨架 (主評估側以高對比力學色塊與角度標籤呈現，對側亦完整繪製手掌三角、拇指與足底三角)
    3. 手部精細結構 (手腕、拇指軸線、食指-小指手掌平面三角形)
    4. 足底支撐結構 (腳踝-腳跟-腳尖支撐三角形)
    """
    out = img.copy()
    overlay = img.copy()

    def to_pt(arr):
        return (int(round(arr[0])), int(round(arr[1])))

    # 取出對側與主側座標
    o_sh, o_el, o_wr = to_pt(pts_dict["o_sh"]), to_pt(pts_dict["o_el"]), to_pt(pts_dict["o_wr"])
    o_th, o_ix, o_pk = to_pt(pts_dict["o_th"]), to_pt(pts_dict["o_ix"]), to_pt(pts_dict["o_pk"])
    o_hp, o_kn, o_an = to_pt(pts_dict["o_hp"]), to_pt(pts_dict["o_kn"]), to_pt(pts_dict["o_an"])
    o_hl, o_ft = to_pt(pts_dict["o_hl"]), to_pt(pts_dict["o_ft"])

    p_ear = to_pt(pts_dict["ear"])
    p_sh  = to_pt(pts_dict["sh"])
    p_el  = to_pt(pts_dict["el"])
    p_wr  = to_pt(pts_dict["wr"])
    p_th  = to_pt(pts_dict["th"])
    p_ix  = to_pt(pts_dict["ix"])
    p_pk  = to_pt(pts_dict["pk"])
    p_hp  = to_pt(pts_dict["hp"])
    p_kn  = to_pt(pts_dict["kn"])
    p_an  = to_pt(pts_dict["an"])
    p_hl  = to_pt(pts_dict["hl"])
    p_ft  = to_pt(pts_dict["ft"])

    # ---- 第一層：半透明力學平面填充 (雙側手掌三角、雙側足底三角、軀幹網格) ----
    o_palm_poly = np.array([o_wr, o_ix, o_pk], dtype=np.int32)
    o_foot_poly = np.array([o_an, o_hl, o_ft], dtype=np.int32)
    palm_poly = np.array([p_wr, p_ix, p_pk], dtype=np.int32)
    foot_poly = np.array([p_an, p_hl, p_ft], dtype=np.int32)
    torso_poly = np.array([p_sh, o_sh, o_hp, p_hp], dtype=np.int32)

    cv2.fillPoly(overlay, [torso_poly], (0, 180, 60))
    cv2.fillPoly(overlay, [o_palm_poly], (255, 200, 0))
    cv2.fillPoly(overlay, [o_foot_poly], (255, 180, 0))
    cv2.fillPoly(overlay, [palm_poly], (0, 220, 255))
    cv2.fillPoly(overlay, [foot_poly], (0, 200, 255))
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    # ---- 第二層：對側完整骨架與關節點 (青藍色系，確保全身雙側皆清晰呈現) ----
    cv2.line(out, p_sh, o_sh, (0, 210, 120), 2, cv2.LINE_AA)
    cv2.line(out, p_hp, o_hp, (0, 210, 120), 2, cv2.LINE_AA)
    cv2.line(out, o_sh, o_hp, (180, 200, 80), 2, cv2.LINE_AA)

    cv2.line(out, o_sh, o_el, (255, 190, 0), 3, cv2.LINE_AA)
    cv2.line(out, o_el, o_wr, (255, 210, 50), 3, cv2.LINE_AA)
    cv2.line(out, o_wr, o_th, (255, 100, 220), 2, cv2.LINE_AA)
    cv2.polylines(out, [o_palm_poly], True, (255, 240, 80), 2, cv2.LINE_AA)

    cv2.line(out, o_hp, o_kn, (255, 170, 50), 3, cv2.LINE_AA)
    cv2.line(out, o_kn, o_an, (255, 190, 80), 3, cv2.LINE_AA)
    cv2.polylines(out, [o_foot_poly], True, (255, 220, 100), 2, cv2.LINE_AA)

    for p_opp in [o_sh, o_el, o_wr, o_th, o_ix, o_pk, o_hp, o_kn, o_an, o_hl, o_ft]:
        cv2.circle(out, p_opp, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, p_opp, 3, (255, 140, 0), -1, cv2.LINE_AA)

    # ---- 第三層：頭部五官 11 點解剖網格 (鼻、雙眼內中外、雙耳、嘴角) ----
    f_pts = [to_pt(pts_dict[f"f_{i}"]) for i in range(11)]
    # 右眼軌跡 (0->4->5->6->8) 與 左眼軌跡 (0->1->2->3->7)
    for a_i, b_i in [(0,4), (4,5), (5,6), (6,8), (0,1), (1,2), (2,3), (3,7), (2,5), (9,10)]:
        cv2.line(out, f_pts[a_i], f_pts[b_i], (0, 255, 255), 2, cv2.LINE_AA)
    mouth_mid = ((f_pts[9][0] + f_pts[10][0]) // 2, (f_pts[9][1] + f_pts[10][1]) // 2)
    cv2.line(out, f_pts[0], mouth_mid, (0, 255, 255), 2, cv2.LINE_AA)

    for fp in f_pts:
        cv2.circle(out, fp, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, fp, 3, (0, 180, 255), -1, cv2.LINE_AA)

    # ---- 第四層：主評估側核心人因力學鏈 (高亮顯示) ----
    cv2.line(out, p_hp, (p_hp[0], max(0, p_sh[1] - 70)), (200, 200, 200), 2, cv2.LINE_AA)

    # 下肢與足底支撐三角形
    cv2.line(out, p_hp, p_kn, (0, 165, 255), 4, cv2.LINE_AA)
    cv2.line(out, p_kn, p_an, (0, 190, 255), 4, cv2.LINE_AA)
    cv2.polylines(out, [foot_poly], True, (0, 235, 255), 3, cv2.LINE_AA)

    # 軀幹與頭頸軸線
    cv2.line(out, p_hp, p_sh, (0, 240, 0), 5, cv2.LINE_AA)
    cv2.line(out, p_sh, p_ear, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.line(out, p_ear, f_pts[0], (0, 215, 255), 3, cv2.LINE_AA)

    # 上肢與手掌力學三角 + 拇指扭轉軸
    cv2.line(out, p_sh, p_el, (0, 60, 255), 5, cv2.LINE_AA)
    cv2.line(out, p_el, p_wr, (0, 140, 255), 5, cv2.LINE_AA)
    cv2.polylines(out, [palm_poly], True, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.line(out, p_wr, p_th, (255, 50, 255), 3, cv2.LINE_AA)

    for pt in [p_ear, p_sh, p_el, p_wr, p_hp, p_kn, p_an]:
        cv2.circle(out, pt, 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)

    for pt in [p_th, p_ix, p_pk, p_hl, p_ft]:
        cv2.circle(out, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pt, 4, (255, 0, 200), -1, cv2.LINE_AA)

    # 標註各關節即時角度與手腕扭轉狀態
    twist_str = "接近旋轉極限(2分)" if wt_score == 2 else "中間範圍(1分)"
    out = put_chinese_text(out, f"頸部:{ang_nk:.0f}°", (p_ear[0] + 12, max(10, p_ear[1] - 12)), (255, 255, 120), 14)
    out = put_chinese_text(out, f"上臂:{ang_ua:.0f}°", (p_sh[0] + 12, p_sh[1] - 8), (0, 255, 255), 14)
    out = put_chinese_text(out, f"前臂:{ang_la:.0f}°", (p_el[0] + 12, p_el[1] + 2), (0, 200, 255), 14)
    out = put_chinese_text(out, f"手腕:{ang_wr:.0f}°|{twist_str}", (p_wr[0] + 12, p_wr[1] + 5), (255, 160, 255), 14)
    out = put_chinese_text(out, f"軀幹:{ang_tk:.0f}°", (p_hp[0] + 12, p_hp[1] - 15), (100, 255, 100), 14)

    if target_side_label:
        h_i, _ = out.shape[:2]
        out = put_chinese_text(out, f"🎥 獨立擷取視角: {target_side_label}專用鏡頭", (12, h_i - 28), (255, 255, 255), 15)

    return out

# ==================== 【正面鏡頭：冠狀面/水平面補正分析 (支援指定單側)】 ====================
def analyze_front_camera(frame, pose_model, target_side):
    """正面影像補正。若核心點品質不足，回報失敗而不是默認所有補正皆為否。"""
    if frame is None or frame.size == 0:
        return False, {}

    orig_h, orig_w = frame.shape[:2]
    scale = min(900 / max(orig_h, orig_w), 1.0)
    frame_resized = cv2.resize(frame, (max(1, int(orig_w * scale)), max(1, int(orig_h * scale))))

    sh_idx = 12 if target_side == "右手側" else 11
    el_idx = 14 if target_side == "右手側" else 13
    wr_idx = 16 if target_side == "右手側" else 15
    required = [11, 12, 13, 14, 15, 16, 23, 24, 7, 8, sh_idx, el_idx, wr_idx]
    results, _, _ = detect_pose_with_fallback(frame_resized, pose_model, required)
    if not results or not getattr(results, "pose_landmarks", None):
        return False, {}

    lm = results.pose_landmarks.landmark
    if not landmarks_meet_visibility(results, required, threshold=0.25):
        return False, {}

    h, w = frame_resized.shape[:2]
    def xy(i):
        return np.array([lm[i].x * w, lm[i].y * h], dtype=float)

    sh_left, sh_right = xy(11), xy(12)
    hp_left, hp_right = xy(23), xy(24)
    midline_x = float(np.mean([sh_left[0], sh_right[0], hp_left[0], hp_right[0]]))
    shoulder_width = max(float(np.linalg.norm(sh_left - sh_right)), 1.0)

    target_shoulder_x = xy(sh_idx)[0]
    elbow_x = xy(el_idx)[0]
    wrist_x = xy(wr_idx)[0]

    # 使用絕對水平位移判斷外展，降低左右鏡像影片造成的符號錯誤。
    is_abducted = abs(elbow_x - target_shoulder_x) > shoulder_width * 0.35
    # 依該側肩膀相對身體中線的位置決定「跨到對側」，不假設右手一定在影像右邊。
    side_direction = np.sign(target_shoulder_x - midline_x)
    is_across_midline = bool(side_direction != 0 and (wrist_x - midline_x) * side_direction < 0)

    # RULA 前臂「跨中線或向外側工作」只加一次分；這裡以外側偏移作為保守代理值，需目視複核。
    elbow_x_from_midline = abs(xy(el_idx)[0] - midline_x)
    wrist_x_from_midline = abs(wrist_x - midline_x)
    is_out_to_side = bool(
        not is_across_midline
        and elbow_x_from_midline > shoulder_width * 0.55
        and wrist_x_from_midline > shoulder_width * 0.65
    )

    shoulder_tilt_ratio = abs(sh_left[1] - sh_right[1]) / shoulder_width
    is_trunk_side_bending = shoulder_tilt_ratio > 0.12

    ear_left, ear_right = xy(7), xy(8)
    ear_tilt_ratio = abs(ear_left[1] - ear_right[1]) / shoulder_width
    is_neck_side_bending = ear_tilt_ratio > 0.15

    mods = {
        "is_abducted": bool(is_abducted),
        "is_across_midline": bool(is_across_midline),
        "is_out_to_side": bool(is_out_to_side),
        "is_side_bending": bool(is_trunk_side_bending),
        "is_neck_side_bending": bool(is_neck_side_bending),
        "front_keypoint_visibility": float(np.mean(_visibility_values(results, required)) * 100.0),
    }
    return True, mods

# ==================== 【側視角鏡頭：單側獨立擷取、分析與 RULA 計算】 ====================
def analyze_side_camera(frame, pose_model, target_side, front_mods, m_val, f_val, manual_mods=None):
    if frame is None or frame.size == 0 or not isinstance(front_mods, dict):
        return False, None, frame, 0.0
    manual_mods = manual_mods or {}
    orig_h, orig_w = frame.shape[:2]
    scale = min(1000 / max(orig_h, orig_w), 1.0)
    work_w, work_h = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
    frame_work = cv2.resize(frame, (work_w, work_h))

    if target_side == "右手側":
        t_ear, t_sh, t_el, t_wr, t_pk, t_ix, t_th = 8, 12, 14, 16, 18, 20, 22
        t_hp, t_kn, t_an, t_hl, t_ft = 24, 26, 28, 30, 32
        o_ear, o_sh, o_el, o_wr, o_pk, o_ix, o_th = 7, 11, 13, 15, 17, 19, 21
        o_hp, o_kn, o_an, o_hl, o_ft = 23, 25, 27, 29, 31
    else:
        t_ear, t_sh, t_el, t_wr, t_pk, t_ix, t_th = 7, 11, 13, 15, 17, 19, 21
        t_hp, t_kn, t_an, t_hl, t_ft = 23, 25, 27, 29, 31
        o_ear, o_sh, o_el, o_wr, o_pk, o_ix, o_th = 8, 12, 14, 16, 18, 20, 22
        o_hp, o_kn, o_an, o_hl, o_ft = 24, 26, 28, 30, 32

    # 只允許評估指定側；不再因對側 visibility 較高就悄悄換手，避免左右手分數標錯。
    required = [t_sh, t_el, t_wr, t_ix, t_pk, t_hp, o_sh, o_hp]
    results, _, _ = detect_pose_with_fallback(frame_work, pose_model, required)
    if not results or not getattr(results, "pose_landmarks", None) or not getattr(results, "pose_world_landmarks", None):
        return False, None, frame_work, 0.0

    n_lm = results.pose_landmarks.landmark
    w_lm = results.pose_world_landmarks.landmark
    if not landmarks_meet_visibility(results, required, threshold=0.25):
        return False, None, frame_work, float(np.mean(_visibility_values(results, required)) * 100.0)
    # 側面常會遮住遠側眼睛／耳朵，改用可見度較高的臉部點，而不是要求兩眼同時可見。
    if max(float(n_lm[t_ear].visibility), float(n_lm[o_ear].visibility)) < 0.25:
        return False, None, frame_work, float(np.mean(_visibility_values(results, required)) * 100.0)
    if max(float(n_lm[2].visibility), float(n_lm[5].visibility)) < 0.25:
        return False, None, frame_work, float(np.mean(_visibility_values(results, required)) * 100.0)

    try:
        def pt2d(idx):
            return np.array([n_lm[idx].x * work_w, n_lm[idx].y * work_h], dtype=float)

        def pt3d(idx):
            return np.array([w_lm[idx].x, w_lm[idx].y, w_lm[idx].z], dtype=float)

        key_vis = _visibility_values(results, required)
        conf_side = float(np.mean(key_vis) * 100.0) if key_vis else 0.0

        # 軀幹中心用雙肩／雙髖加權，耳朵選可見度較高的一側。
        sh = pt2d(t_sh) * 0.65 + pt2d(o_sh) * 0.35
        hp = pt2d(t_hp) * 0.65 + pt2d(o_hp) * 0.35
        ear = pt2d(t_ear) if n_lm[t_ear].visibility >= n_lm[o_ear].visibility else pt2d(o_ear)
        eye_idx = 2 if n_lm[2].visibility >= n_lm[5].visibility else 5
        visible_eye = pt2d(eye_idx)

        # 目標側上肢與手掌點；不以對側手臂替換目標側。
        el, wr, th, ix, pk = pt2d(t_el), pt2d(t_wr), pt2d(t_th), pt2d(t_ix), pt2d(t_pk)
        b_el, b_wr, b_th, b_ix, b_pk = pt2d(o_el), pt2d(o_wr), pt2d(o_th), pt2d(o_ix), pt2d(o_pk)
        w3_el, w3_wr, w3_th, w3_ix, w3_pk = pt3d(t_el), pt3d(t_wr), pt3d(t_th), pt3d(t_ix), pt3d(t_pk)

        # 腿部點只用於畫圖，不再用雙腳影像高度差自動代替 RULA 的支撐／平衡判斷。
        kn, an, hl, ft = pt2d(t_kn), pt2d(t_an), pt2d(t_hl), pt2d(t_ft)
        b_kn, b_an, b_hl, b_ft = pt2d(o_kn), pt2d(o_an), pt2d(o_hl), pt2d(o_ft)

        torso_len = np.linalg.norm(hp - sh)

        # ==================== 【RULA 獨立關節角度解算】 ====================
        ang_lower_arm = abs(180.0 - calculate_2d_angle(sh, el, wr))
        
        trunk_vec = hp - sh
        ref_sh_down = sh + trunk_vec
        ang_upper_arm = calculate_2d_angle(ref_sh_down, sh, el)
        
        palm_mid_2d = (ix + pk) * 0.5
        palm_len_2d = np.linalg.norm(palm_mid_2d - wr)
        raw_wrist_2d = abs(180.0 - calculate_2d_angle(el, wr, palm_mid_2d)) if palm_len_2d > 6.0 else float("nan")
        palm_mid_3d = (w3_ix + w3_pk) * 0.5
        raw_wrist_3d = abs(180.0 - calculate_3d_angle(w3_el, w3_wr, palm_mid_3d))
        if not np.isfinite(raw_wrist_2d):
            ang_wrist = raw_wrist_3d
        else:
            ang_wrist = float(raw_wrist_2d * 0.6 + raw_wrist_3d * 0.4)
        ang_wrist = float(np.clip(ang_wrist, 0.0, 60.0))

        # MediaPipe Pose 的拇指／手腕三維夾角不是前臂旋前旋後角，因此腕扭轉與偏移改由使用者目視確認。
        wt_score = int(manual_mods.get("wrist_twist_score", 1))
        is_wrist_deviated = bool(manual_mods.get("wrist_deviated", False))

        vertical_up = np.array([hp[0], hp[1] - 100.0])
        ang_trunk = calculate_2d_angle(vertical_up, hp, sh)

        head_center = ear * 0.6 + visible_eye * 0.4
        ang_neck = abs(180.0 - calculate_2d_angle(hp, sh, head_center))

        # RULA 腿足支撐與平衡需依實際作業情境確認，不再用兩腳高度差猜測。
        lg_score = int(manual_mods.get("leg_score", 2))
        if not np.isfinite(ang_lower_arm) or not np.isfinite(ang_upper_arm) or not np.isfinite(ang_trunk) or not np.isfinite(ang_neck) or not np.isfinite(ang_wrist):
            return False, None, frame_work, conf_side

        # RULA 查表計算；不可由單一 2D 側面可靠辨識的條件由使用者補正。
        s_ua = get_upper_arm_score(
            ang_upper_arm,
            is_abducted=front_mods.get("is_abducted", False),
            is_raised=manual_mods.get("shoulder_raised", False),
            is_supported=manual_mods.get("arm_supported", False),
            is_extended=manual_mods.get("upper_arm_extended", False),
        )
        s_la = get_lower_arm_score(
            ang_lower_arm,
            is_across_midline=front_mods.get("is_across_midline", False),
            is_out_to_side=front_mods.get("is_out_to_side", False),
        )
        s_wr = get_wrist_score(ang_wrist, is_deviated=is_wrist_deviated)
        s_tk = get_trunk_score(
            ang_trunk,
            is_twisted=manual_mods.get("trunk_twisted", False),
            is_side_bending=front_mods.get("is_side_bending", False),
            is_extended=manual_mods.get("trunk_extended", False),
        )
        s_nk = get_neck_score(
            ang_neck,
            is_twisted=manual_mods.get("neck_twisted", False),
            is_side_bending=front_mods.get("is_neck_side_bending", False),
            is_extended=manual_mods.get("neck_extended", False),
        )
        
        sc_a, sc_b, sc_c, sc_d, grand, al = compute_rula_full(
            s_ua, s_la, s_wr, wt_score, s_nk, s_tk, lg_score, m_val, f_val
        )
        
        # 打包全部 33 點座標供繪圖器渲染
        pts_dict = {
            "ear": ear, "sh": sh, "el": el, "wr": wr, "th": th, "ix": ix, "pk": pk,
            "hp": hp, "kn": kn, "an": an, "hl": hl, "ft": ft,
            "o_sh": pt2d(o_sh), "o_el": b_el, "o_wr": b_wr, "o_th": b_th, "o_ix": b_ix, "o_pk": b_pk,
            "o_hp": pt2d(o_hp), "o_kn": b_kn, "o_an": b_an, "o_hl": b_hl, "o_ft": b_ft
        }
        for f_i in range(11):
            pts_dict[f"f_{f_i}"] = pt2d(f_i)

        frame_annotated = draw_full_biomechanical_skeleton(
            frame_work, pts_dict,
            ang_upper_arm, ang_lower_arm, ang_wrist, ang_trunk, ang_neck, wt_score,
            target_side_label=target_side
        )
        
        side_prefix = "右手" if target_side == "右手側" else "左手"
        data = {
            "頸部角度": round(ang_neck, 1),
            "軀幹角度": round(ang_trunk, 1),
            f"{side_prefix}上臂角度": round(ang_upper_arm, 1),
            f"{side_prefix}前臂角度": round(ang_lower_arm, 1),
            f"{side_prefix}手腕角度": round(ang_wrist, 1),
            f"{side_prefix}手腕扭轉": "接近旋轉極限(2分)" if wt_score == 2 else "中間範圍(1分；人工確認)",
            f"{side_prefix}手腕偏移補正": "是 (+1分)" if is_wrist_deviated else "否（人工確認）",
            "足底支撐狀態": "未受支撐或不平衡(2分；人工確認)" if lg_score == 2 else "受支撐且平衡(1分；人工確認)",
            f"{side_prefix}上臂外展補正": "是 (+1分)" if front_mods.get("is_abducted") else "否",
            f"{side_prefix}肩部抬高補正": "是 (+1分)" if manual_mods.get("shoulder_raised") else "否（人工確認）",
            f"{side_prefix}上臂支撐／倚靠補正": "是 (-1分)" if manual_mods.get("arm_supported") else "否（人工確認）",
            f"{side_prefix}上臂向後伸展補正": "是（基礎分至少 2）" if manual_mods.get("upper_arm_extended") else "否（人工確認）",
            f"{side_prefix}橫越中線補正": "是 (+1分)" if front_mods.get("is_across_midline") else "否",
            f"{side_prefix}前臂向外側工作補正": "是 (+1分)" if front_mods.get("is_out_to_side") else "否（影像代理判斷）",
            "軀幹側彎補正": "是 (+1分)" if front_mods.get("is_side_bending") else "否（影像代理判斷）",
            "軀幹扭轉補正": "是 (+1分)" if manual_mods.get("trunk_twisted") else "否（人工確認）",
            "軀幹向後伸展": "是（軀幹基礎分 2）" if manual_mods.get("trunk_extended") else "否（人工確認）",
            "頸部扭轉補正": "是 (+1分)" if manual_mods.get("neck_twisted") else "否（人工確認）",
            "頸部後仰／伸展": "是（頸部基礎分 4）" if manual_mods.get("neck_extended") else "否（人工確認）",
            f"{side_prefix} Score A": sc_a,
            "頸軀腿 Score B": sc_b,
            f"{side_prefix} Score C": sc_c,
            f"{side_prefix} Grand Score": grand,
            f"{side_prefix} AL": al,
            "最危害側": target_side,
            "最終最高分": grand,
            "最終 AL": al
        }
        return True, data, frame_annotated, conf_side
    except Exception:
        logging.exception("analyze_side_camera failed for %s", target_side)
        return False, None, frame_work, 0.0


# ==================== 【Streamlit 網頁應用介面】 ====================
st.set_page_config(page_title="RULA 三視角人因工程 AI 評估系統", layout="wide")
st.title("RULA 姿勢危害評估系統")
st.caption(f"部署版本：{APP_BUILD_TAG}｜執行檔：{os.path.basename(__file__)}｜MediaPipe Pose complexity={POSE_MODEL_COMPLEXITY}")
st.markdown(
    "> **系統說明**：本工具以 MediaPipe Pose 關鍵點估算部分姿勢角度，遇到關鍵點可見度不足時會嘗試原圖、對比增強及放大圖；仍不可靠的樣本會略過。RULA 的手腕扭轉、腿足支撐、聳肩、上臂支撐及扭轉等項目需人工確認，輸出仍須由受訓評估者複核，不應單獨作為醫療或職安決策依據。"
)

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.df_records = None
    st.session_state.df_right_only = None
    st.session_state.df_left_only = None
    st.session_state.display_images_b64 = []
    st.session_state.display_right_b64 = []
    st.session_state.display_left_b64 = []
    st.session_state.zip_bytes = None
    st.session_state.zip_filename = "RULA_Report.zip"
    st.session_state.analyzed_mode = ""

# 讓使用者選擇單側評估，或三鏡頭左右同步分開計算
eval_mode = st.radio(
    "請選擇本次作業分析之【評估模式】:",
    options=[
        "右手側",
        "左手側",
        "左右側同步計算 (需上傳正面、右手側、左手側三個視角影片)"
    ],
    index=0,
    horizontal=True,
    help="選取單側時需上傳「正面 + 該側鏡頭」；選取「左右側同步計算」時，三個鏡頭影片皆須上傳，系統將針對左右手分開計算、分析與擷取。"
)

st.header("第一步：上傳視角影片")
col_front, col_right, col_left = st.columns(3)

with col_front:
    vid_front_file = st.file_uploader("1. 正面鏡頭 (必填: 判定外展/交叉/側傾)", type=['mp4', 'mov', 'avi'])
with col_right:
    vid_right_file = st.file_uploader("2. 右手側鏡頭 (量測並擷取右手側動作)", type=['mp4', 'mov', 'avi'])
with col_left:
    vid_left_file = st.file_uploader("3. 左手側鏡頭 (量測並擷取左手側動作)", type=['mp4', 'mov', 'avi'])

st.header("第二步：等距抽樣與負載參數設定")
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    num_samples = st.slider("等距抽樣張數", min_value=5, max_value=100, value=25)
with col_p2:
    combo_muscle = st.selectbox("肌肉使用分數 (Muscle Score):", ["0: 非靜態且動作少於每分鐘 4 次", "1: 姿勢維持超過 1 分鐘或動作每分鐘至少 4 次"])
with col_p3:
    combo_force = st.selectbox("負載與施力 (Force/Load):", ["0: < 2kg (間歇施力)", "1: 2-10kg (間歇施力)", "2: 2-10kg 靜態／重複施力", "3: >10kg 負載、重複施力或衝擊力"])


st.header("第三步：人工確認無法由單一影像可靠判斷的 RULA 條件")
st.caption("以下補正會套用到整段影片的每個抽樣點；若姿勢在影片中變化很大，請逐張複核結果。腿足支撐與手腕扭轉不再由錯誤的影像幾何猜測。")
mod_col_a, mod_col_b, mod_col_c = st.columns(3)
with mod_col_a:
    leg_support_choice = st.selectbox(
        "腿與足部支撐／平衡（必選）",
        ["請先目視確認…", "腿足受支撐且平衡（1分）", "腿足未受支撐或不平衡（2分）"],
        index=0,
        help="RULA 腿部評分需要判斷是否有支撐且平衡；單靠側面關鍵點高度無法可靠判斷。"
    )
    neck_twisted_manual = st.checkbox("頸部有扭轉（+1）", value=False)
    neck_extended_manual = st.checkbox("頸部後仰／伸展（頸部基礎分 4）", value=False)
    trunk_twisted_manual = st.checkbox("軀幹有扭轉（+1）", value=False)
    trunk_extended_manual = st.checkbox("軀幹向後伸展（軀幹基礎分 2）", value=False)
with mod_col_b:
    right_wrist_twist_choice = st.selectbox("右手腕扭轉（必選）", ["請先目視確認…", "中間範圍（1分）", "接近旋轉極限（2分）"], index=0)
    right_wrist_deviated = st.checkbox("右手腕偏離中線／尺橈偏移（+1）", value=False)
    right_shoulder_raised = st.checkbox("右肩抬高／聳肩（上臂 +1）", value=False)
    right_arm_supported = st.checkbox("右上臂有支撐或身體倚靠（上臂 -1）", value=False)
    right_arm_extended = st.checkbox("右上臂向身體後方伸展超過 20°", value=False)
with mod_col_c:
    left_wrist_twist_choice = st.selectbox("左手腕扭轉（必選）", ["請先目視確認…", "中間範圍（1分）", "接近旋轉極限（2分）"], index=0)
    left_wrist_deviated = st.checkbox("左手腕偏離中線／尺橈偏移（+1）", value=False)
    left_shoulder_raised = st.checkbox("左肩抬高／聳肩（上臂 +1）", value=False)
    left_arm_supported = st.checkbox("左上臂有支撐或身體倚靠（上臂 -1）", value=False)
    left_arm_extended = st.checkbox("左上臂向身體後方伸展超過 20°", value=False)


def build_manual_mods(side_name):
    is_right = side_name == "右手側"
    twist_choice = right_wrist_twist_choice if is_right else left_wrist_twist_choice
    return {
        "leg_score": 1 if leg_support_choice == "腿足受支撐且平衡（1分）" else 2,
        "wrist_twist_score": 1 if twist_choice == "中間範圍（1分）" else 2,
        "wrist_deviated": right_wrist_deviated if is_right else left_wrist_deviated,
        "shoulder_raised": right_shoulder_raised if is_right else left_shoulder_raised,
        "arm_supported": right_arm_supported if is_right else left_arm_supported,
        "upper_arm_extended": right_arm_extended if is_right else left_arm_extended,
        "neck_twisted": neck_twisted_manual,
        "neck_extended": neck_extended_manual,
        "trunk_twisted": trunk_twisted_manual,
        "trunk_extended": trunk_extended_manual,
    }

# 執行分析按鈕
if st.button("🚀 啟動全關節生物力學分析", type="primary", use_container_width=True):
    is_dual_mode = eval_mode.startswith("左右側同步計算")

    # 嚴格檢查影片上傳完整性
    if is_dual_mode:
        if not (vid_front_file and vid_right_file and vid_left_file):
            st.error("❌ 您選擇了【左右側同步計算】，請務必同時上傳「1. 正面鏡頭」、「2. 右手側鏡頭」與「3. 左手側鏡頭」三個視角的影片！")
            st.stop()
    else:
        selected_side_vid = vid_right_file if eval_mode == "右手側" else vid_left_file
        if not vid_front_file:
            st.error("❌ 缺少【正面鏡頭影片】！正面鏡頭是用於校正水平面外展與中線穿越的必要基準。")
            st.stop()
        elif not selected_side_vid:
            st.error(f"❌ 缺少【{eval_mode}鏡頭影片】！您選擇評估「{eval_mode}」，必須提供該側面的拍攝視角進行主屈曲量測。")
            st.stop()

    required_manual_sides = ["右手側", "左手側"] if is_dual_mode else [eval_mode]
    if leg_support_choice == "請先目視確認…":
        st.error("請先人工確認腿與足部是否受支撐且平衡，再啟動分析。")
        st.stop()
    for required_side in required_manual_sides:
        twist_choice = right_wrist_twist_choice if required_side == "右手側" else left_wrist_twist_choice
        if twist_choice == "請先目視確認…":
            st.error(f"請先人工確認{required_side}手腕扭轉狀態，再啟動分析。")
            st.stop()

    with st.spinner(f"系統正在進行 33 點全解剖特徵擷取與 RULA 解算（模式：{eval_mode}）..."):
        st.session_state.display_images_b64 = []
        st.session_state.display_right_b64 = []
        st.session_state.display_left_b64 = []
        st.session_state.zip_bytes = None
        gc.collect()

        m_val = int(combo_muscle.split(":")[0])
        f_val = int(combo_force.split(":")[0])

        # ==================== 【模式 A：左右側同步計算 (三鏡頭分開計算、分析與擷取)】 ====================
        if is_dual_mode:
            t_front = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_front.write(vid_front_file.getvalue()); t_front.close()

            t_right = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_right.write(vid_right_file.getvalue()); t_right.close()

            t_left = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_left.write(vid_left_file.getvalue()); t_left.close()

            cap_front = cv2.VideoCapture(t_front.name)
            cap_right = cv2.VideoCapture(t_right.name)
            cap_left  = cv2.VideoCapture(t_left.name)

            fps_front = float(cap_front.get(cv2.CAP_PROP_FPS) or 30.0)
            fps_right = float(cap_right.get(cv2.CAP_PROP_FPS) or 30.0)
            fps_left = float(cap_left.get(cv2.CAP_PROP_FPS) or 30.0)
            count_front = int(cap_front.get(cv2.CAP_PROP_FRAME_COUNT))
            count_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
            count_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_front = fps_front if np.isfinite(fps_front) and fps_front > 0 else 30.0
            fps_right = fps_right if np.isfinite(fps_right) and fps_right > 0 else 30.0
            fps_left = fps_left if np.isfinite(fps_left) and fps_left > 0 else 30.0
            durations = [count_front / fps_front, count_right / fps_right, count_left / fps_left]
            duration = min(durations) if all(d > 0 for d in durations) else 0.0

            if duration <= 0:
                for c in [cap_front, cap_right, cap_left]: c.release()
                for p in [t_front.name, t_right.name, t_left.name]:
                    try: os.remove(p)
                    except Exception: pass
                st.error("影片解析長度為 0，請檢查影片編碼格式！")
                st.stop()

            # 以共同時間戳取樣，而非把不同 FPS 影片硬套到相同 frame index。
            sample_times = np.linspace(0, max(0.0, duration - 1.0 / max(fps_front, fps_right, fps_left)), num_samples)

            combined_records = []
            right_only_records = []
            left_only_records = []
            zip_right_imgs = []
            zip_left_imgs = []
            display_right_b64 = []
            display_left_b64 = []
            skipped_right = 0
            skipped_left = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            with mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=POSE_MODEL_COMPLEXITY,
                min_detection_confidence=0.50
            ) as pose:
                total_samples = len(sample_times)
                for i, sample_time in enumerate(sample_times):
                    sec = round(float(sample_time), 2)
                    idx_front = min(count_front - 1, max(0, int(round(sample_time * fps_front))))
                    idx_right = min(count_right - 1, max(0, int(round(sample_time * fps_right))))
                    idx_left = min(count_left - 1, max(0, int(round(sample_time * fps_left))))
                    cap_front.set(cv2.CAP_PROP_POS_FRAMES, idx_front)
                    cap_right.set(cv2.CAP_PROP_POS_FRAMES, idx_right)
                    cap_left.set(cv2.CAP_PROP_POS_FRAMES, idx_left)

                    s_f, f_frame = cap_front.read()
                    s_r, r_frame = cap_right.read()
                    s_l, l_frame = cap_left.read()

                    if s_f and s_r and s_l:
                        # 1. 正面鏡頭分別提取右手與左手的冠狀面外展/交叉補正
                        front_ok_r, front_mods_r = analyze_front_camera(f_frame, pose, "右手側")
                        front_ok_l, front_mods_l = analyze_front_camera(f_frame, pose, "左手側")
                        if not front_ok_r:
                            front_mods_r = None
                        if not front_ok_l:
                            front_mods_l = None

                        # 2. 右側鏡頭只評估右側關鍵點；低品質正面補正會使該樣本略過，不默認為「否」。
                        succ_r, data_r, img_r, conf_r = analyze_side_camera(
                            r_frame, pose, "右手側", front_mods_r, m_val, f_val,
                            manual_mods=build_manual_mods("右手側")
                        )

                        # 3. 左側鏡頭只評估左側關鍵點。
                        succ_l, data_l, img_l, conf_l = analyze_side_camera(
                            l_frame, pose, "左手側", front_mods_l, m_val, f_val,
                            manual_mods=build_manual_mods("左手側")
                        )
                        if not succ_r or data_r is None:
                            skipped_right += 1
                        if not succ_l or data_l is None:
                            skipped_left += 1

                        if succ_r and data_r is not None:
                            rec_r = {"樣本編號": i + 1, "時間(秒)": sec, "評估側": "右手側", "右側信賴度(%)": round(conf_r, 1)}
                            rec_r.update(data_r)
                            right_only_records.append(rec_r)

                            img_r = put_chinese_text(img_r, f"時間: {sec}s | 獨立擷取: 右手側", (15, 15), (0, 255, 255), 20)
                            img_r = put_chinese_text(img_r, f"右手 Grand Score: {data_r['右手 Grand Score']} 分 ({data_r['右手 AL']})", (15, 45), (0, 0, 255) if data_r['右手 Grand Score'] > 4 else (0, 255, 0), 20)
                            img_r = put_chinese_text(img_r, f"外展:{data_r['右手上臂外展補正']} | 交叉:{data_r['右手橫越中線補正']}", (15, 75), (255, 180, 0), 16)

                            oh, ow = img_r.shape[:2]
                            out_scale = min(720 / max(oh, ow), 1.0)
                            if out_scale < 1.0:
                                img_r = cv2.resize(img_r, (int(ow * out_scale), int(oh * out_scale)))
                            ok_r, buf_r = cv2.imencode(".jpg", img_r, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            if ok_r:
                                b_r = buf_r.tobytes()
                                zip_right_imgs.append((f"Sample_{i+1:03d}_{sec}s_Right.jpg", b_r))
                                cap_str_r = f"右手樣本 #{i+1} ({sec}s) | 總分: {data_r['右手 Grand Score']} ({data_r['右手 AL']}) | 手腕: {data_r['右手手腕角度']}°"
                                display_right_b64.append((cap_str_r, base64.b64encode(b_r).decode("utf-8")))

                        if succ_l and data_l is not None:
                            rec_l = {"樣本編號": i + 1, "時間(秒)": sec, "評估側": "左手側", "左側信賴度(%)": round(conf_l, 1)}
                            rec_l.update(data_l)
                            left_only_records.append(rec_l)

                            img_l = put_chinese_text(img_l, f"時間: {sec}s | 獨立擷取: 左手側", (15, 15), (0, 255, 255), 20)
                            img_l = put_chinese_text(img_l, f"左手 Grand Score: {data_l['左手 Grand Score']} 分 ({data_l['左手 AL']})", (15, 45), (0, 0, 255) if data_l['左手 Grand Score'] > 4 else (0, 255, 0), 20)
                            img_l = put_chinese_text(img_l, f"外展:{data_l['左手上臂外展補正']} | 交叉:{data_l['左手橫越中線補正']}", (15, 75), (255, 180, 0), 16)

                            oh, ow = img_l.shape[:2]
                            out_scale = min(720 / max(oh, ow), 1.0)
                            if out_scale < 1.0:
                                img_l = cv2.resize(img_l, (int(ow * out_scale), int(oh * out_scale)))
                            ok_l, buf_l = cv2.imencode(".jpg", img_l, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            if ok_l:
                                b_l = buf_l.tobytes()
                                zip_left_imgs.append((f"Sample_{i+1:03d}_{sec}s_Left.jpg", b_l))
                                cap_str_l = f"左手樣本 #{i+1} ({sec}s) | 總分: {data_l['左手 Grand Score']} ({data_l['左手 AL']}) | 手腕: {data_l['左手手腕角度']}°"
                                display_left_b64.append((cap_str_l, base64.b64encode(b_l).decode("utf-8")))

                        # 若該秒左右手皆成功獨立解算，建立雙側對照總表紀錄
                        if succ_r and succ_l and data_r is not None and data_l is not None:
                            r_g = data_r["右手 Grand Score"]
                            l_g = data_l["左手 Grand Score"]
                            worst_side = "右手側" if r_g >= l_g else "左手側"
                            final_grand = max(r_g, l_g)
                            final_al = data_r["右手 AL"] if r_g >= l_g else data_l["左手 AL"]
                            overall_conf = round((conf_r + conf_l) / 2.0, 1)

                            comb_row = {
                                "樣本編號": i + 1,
                                "時間(秒)": sec,
                                "頸部角度": round((data_r["頸部角度"] + data_l["頸部角度"]) / 2.0, 1),
                                "軀幹角度": round((data_r["軀幹角度"] + data_l["軀幹角度"]) / 2.0, 1),
                                "右上臂角度": data_r["右手上臂角度"],
                                "右前臂角度": data_r["右手前臂角度"],
                                "右手腕角度": data_r["右手手腕角度"],
                                "右手腕扭轉": data_r["右手手腕扭轉"],
                                "左上臂角度": data_l["左手上臂角度"],
                                "左前臂角度": data_l["左手前臂角度"],
                                "左手腕角度": data_l["左手手腕角度"],
                                "左手腕扭轉": data_l["左手手腕扭轉"],
                                "右手 Score A": data_r["右手 Score A"],
                                "右手 Score C": data_r["右手 Score C"],
                                "右手 Grand Score": r_g,
                                "右手 AL": data_r["右手 AL"],
                                "左手 Score A": data_l["左手 Score A"],
                                "左手 Score C": data_l["左手 Score C"],
                                "左手 Grand Score": l_g,
                                "左手 AL": data_l["左手 AL"],
                                "最危害側": worst_side,
                                "最終最高分": final_grand,
                                "最終 AL": final_al,
                                "Overall_Conf": overall_conf
                            }
                            combined_records.append(comb_row)
                    else:
                        skipped_right += 1
                        skipped_left += 1

                    progress_bar.progress((i + 1) / total_samples)
                    status_text.text(f"左右側獨立擷取與姿態解算進度: {i+1} / {total_samples} 筆完成")

            for c in [cap_front, cap_right, cap_left]: c.release()
            for p in [t_front.name, t_right.name, t_left.name]:
                try: os.remove(p)
                except Exception: pass

            df_comb = pd.DataFrame(combined_records) if combined_records else pd.DataFrame()
            df_r = pd.DataFrame(right_only_records) if right_only_records else pd.DataFrame()
            df_l = pd.DataFrame(left_only_records) if left_only_records else pd.DataFrame()

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                if not df_comb.empty:
                    zf.writestr("RULA_左右同步彙整報表.csv", df_comb.to_csv(index=False).encode('utf-8-sig'))
                if not df_r.empty:
                    zf.writestr("RULA_右手側獨立報表.csv", df_r.to_csv(index=False).encode('utf-8-sig'))
                if not df_l.empty:
                    zf.writestr("RULA_左手側獨立報表.csv", df_l.to_csv(index=False).encode('utf-8-sig'))
                for fname, fbytes in zip_right_imgs:
                    zf.writestr(f"右側鏡頭擷取照片/{fname}", fbytes)
                for fname, fbytes in zip_left_imgs:
                    zf.writestr(f"左側鏡頭擷取照片/{fname}", fbytes)

            fixed_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.session_state.analysis_done = True
            st.session_state.df_records = df_comb
            st.session_state.df_right_only = df_r
            st.session_state.df_left_only = df_l
            st.session_state.display_right_b64 = display_right_b64
            st.session_state.display_left_b64 = display_left_b64
            st.session_state.zip_bytes = zip_buffer.getvalue()
            st.session_state.zip_filename = f"RULA_DualSide_Separate_Report_{fixed_time_str}.zip"
            st.session_state.analyzed_mode = "左右側同步計算"
            st.session_state.skipped_info = {"expected": total_samples, "right": skipped_right, "left": skipped_left}
            gc.collect()

        # ==================== 【模式 B：單側評估 (右手側 或 左手側)】 ====================
        else:
            target_side = eval_mode
            selected_side_vid = vid_right_file if target_side == "右手側" else vid_left_file

            t_front = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_front.write(vid_front_file.getvalue()); t_front.close()
            
            t_side = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_side.write(selected_side_vid.getvalue()); t_side.close()

            cap_front = cv2.VideoCapture(t_front.name)
            cap_side = cv2.VideoCapture(t_side.name)

            fps_front = float(cap_front.get(cv2.CAP_PROP_FPS) or 30.0)
            fps_side = float(cap_side.get(cv2.CAP_PROP_FPS) or 30.0)
            count_front = int(cap_front.get(cv2.CAP_PROP_FRAME_COUNT))
            count_side = int(cap_side.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_front = fps_front if np.isfinite(fps_front) and fps_front > 0 else 30.0
            fps_side = fps_side if np.isfinite(fps_side) and fps_side > 0 else 30.0
            durations = [count_front / fps_front, count_side / fps_side]
            duration = min(durations) if all(d > 0 for d in durations) else 0.0

            if duration <= 0:
                cap_front.release(); cap_side.release()
                try:
                    os.remove(t_front.name); os.remove(t_side.name)
                except OSError:
                    pass
                st.error("影片解析長度為 0，請檢查影片編碼格式！")
                st.stop()

            sample_times = np.linspace(0, max(0.0, duration - 1.0 / max(fps_front, fps_side)), num_samples)

            records = []
            zip_images = []
            display_images_b64 = []
            skipped_count = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            with mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=POSE_MODEL_COMPLEXITY,
                min_detection_confidence=0.50
            ) as pose:
                total_samples = len(sample_times)
                for i, sample_time in enumerate(sample_times):
                    sec = round(float(sample_time), 2)
                    idx_front = min(count_front - 1, max(0, int(round(sample_time * fps_front))))
                    idx_side = min(count_side - 1, max(0, int(round(sample_time * fps_side))))
                    cap_front.set(cv2.CAP_PROP_POS_FRAMES, idx_front)
                    cap_side.set(cv2.CAP_PROP_POS_FRAMES, idx_side)
                    s_f, f_frame = cap_front.read()
                    s_s, s_frame = cap_side.read()

                    if s_f and s_s:
                        front_ok, front_mods = analyze_front_camera(f_frame, pose, target_side)
                        if not front_ok:
                            front_mods = None
                        succ_s, data, side_annotated, conf = analyze_side_camera(
                            s_frame, pose, target_side, front_mods, m_val, f_val,
                            manual_mods=build_manual_mods(target_side)
                        )
                        if not succ_s or data is None:
                            skipped_count += 1

                        if succ_s and data is not None:
                            record = {
                                "樣本編號": i + 1,
                                "時間(秒)": sec,
                                "主要評估側": target_side,
                                "視角信賴度(%)": round(conf, 1)
                            }
                            record.update(data)
                            records.append(record)

                            side_annotated = put_chinese_text(side_annotated, f"時間: {sec}s | 評估側: {target_side}", (15, 15), (0, 255, 255), 20)
                            side_annotated = put_chinese_text(side_annotated, f"Grand Score: {data['最終最高分']} 分 ({data['最終 AL']})", (15, 45), (0, 0, 255) if data['最終最高分'] > 4 else (0, 255, 0), 20)
                            side_annotated = put_chinese_text(side_annotated, f"外展:{data[f'{target_side[:2]}上臂外展補正']} | 交叉:{data[f'{target_side[:2]}橫越中線補正']}", (15, 75), (255, 180, 0), 16)

                            oh, ow = side_annotated.shape[:2]
                            out_scale = min(720 / max(oh, ow), 1.0)
                            if out_scale < 1.0:
                                side_annotated = cv2.resize(side_annotated, (int(ow * out_scale), int(oh * out_scale)))

                            is_ok, buf = cv2.imencode(".jpg", side_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                            if is_ok:
                                jpg_bytes = buf.tobytes()
                                side_en = "Right" if target_side == "右手側" else "Left"
                                zip_images.append((f"Sample_{i+1:03d}_{sec}s_{side_en}.jpg", jpg_bytes))
                                
                                b64_img = base64.b64encode(jpg_bytes).decode("utf-8")
                                caption_str = f"樣本 #{i+1} ({sec}s) | 總分: {data['最終最高分']} ({data['最終 AL']}) | 手腕: {data[f'{target_side[:2]}手腕角度']}°"
                                display_images_b64.append((caption_str, b64_img))
                    else:
                        skipped_count += 1

                    progress_bar.progress((i + 1) / total_samples)
                    status_text.text(f"全關節姿態解算進度: {i+1} / {total_samples} 筆完成")

            cap_front.release(); cap_side.release()
            try:
                os.remove(t_front.name); os.remove(t_side.name)
            except Exception:
                pass

            df_result = pd.DataFrame(records) if records else pd.DataFrame()
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                if not df_result.empty:
                    zf.writestr("RULA_Report.csv", df_result.to_csv(index=False).encode('utf-8-sig'))
                for fname, fbytes in zip_images:
                    zf.writestr(f"Photos/{fname}", fbytes)

            side_en_tag = "Right" if target_side == "右手側" else "Left"
            fixed_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.session_state.analysis_done = True
            st.session_state.df_records = df_result
            st.session_state.display_images_b64 = display_images_b64
            st.session_state.zip_bytes = zip_buffer.getvalue()
            st.session_state.zip_filename = f"RULA_{side_en_tag}_Report_{fixed_time_str}.zip"
            st.session_state.analyzed_mode = target_side
            st.session_state.skipped_info = {"expected": total_samples, "single": skipped_count}
            gc.collect()

# ==================== 【網頁即時成果展示與打包下載區】 ====================
if st.session_state.analysis_done:
    st.divider()
    mode_now = st.session_state.analyzed_mode
    st.success(f"✅ 【{mode_now}】姿態分析流程完成！以下為即時分析結果：")
    skipped_info = st.session_state.get("skipped_info", {})
    if mode_now == "左右側同步計算":
        skipped_r = int(skipped_info.get("right", 0))
        skipped_l = int(skipped_info.get("left", 0))
        if skipped_r or skipped_l:
            st.warning(f"有樣本因影片讀取失敗或關鍵點品質不足而略過：右側 {skipped_r}/{skipped_info.get('expected', '?')}，左側 {skipped_l}/{skipped_info.get('expected', '?')}。略過的樣本不會自動補成低分，請調整拍攝或改用人工評估。")
    else:
        skipped_n = int(skipped_info.get("single", 0))
        if skipped_n:
            st.warning(f"有 {skipped_n}/{skipped_info.get('expected', '?')} 個抽樣點因影片讀取失敗或關鍵點品質不足而略過；略過的樣本不會自動補成低分，請調整拍攝或改用人工評估。")
    
    df_show = st.session_state.df_records
    
    if df_show is not None and not df_show.empty:
        # 1. 核心人因指標看板
        st.subheader("📈 RULA 危害指標總覽")
        st.caption("視角信賴度為本系統所用關鍵點的平均 visibility 指標，並非 RULA 正確率或經驗證的統計信賴區間。請同時檢查逐張標註影像及漏判樣本。")
        avg_score = df_show["最終最高分"].mean()
        max_score = df_show["最終最高分"].max()
        high_risk_pct = (df_show["最終最高分"] >= 5).mean() * 100

        if mode_now == "左右側同步計算":
            r_avg = df_show["右手 Grand Score"].mean()
            l_avg = df_show["左手 Grand Score"].mean()
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("最終平均最高分", f"{avg_score:.2f} 分")
            m2.metric("右手獨立平均總分", f"{r_avg:.2f} 分")
            m3.metric("左手獨立平均總分", f"{l_avg:.2f} 分")
            m4.metric("最高風險峰值", f"{max_score} 分")
            m5.metric("高危害佔比 (AL3/AL4)", f"{high_risk_pct:.1f}%")
        else:
            avg_conf = df_show["視角信賴度(%)"].mean()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("平均 RULA 總分", f"{avg_score:.2f} 分")
            m2.metric("最高風險峰值", f"{max_score} 分")
            m3.metric("高危害佔比 (AL3/AL4)", f"{high_risk_pct:.1f}%")
            m4.metric("平均視角信賴度", f"{avg_conf:.1f}%")

        # 2. 互動式數據報表
        if mode_now == "左右側同步計算":
            st.subheader("📊 詳細抽樣數據報表（可切換檢視：雙側對照總表 / 右手獨立表 / 左手獨立表）")
            tab_comb, tab_r, tab_l = st.tabs(["📋 左右雙側對照總表", "👉 右手側獨立分析表", "👈 左手側獨立分析表"])
            with tab_comb:
                st.dataframe(df_show, use_container_width=True)
            with tab_r:
                st.dataframe(st.session_state.df_right_only, use_container_width=True)
            with tab_l:
                st.dataframe(st.session_state.df_left_only, use_container_width=True)
        else:
            st.subheader("📊 詳細抽樣數據報表")
            st.dataframe(df_show, use_container_width=True)

        # 3. 瀏覽器本地直出 ZIP 下載按鈕
        st.subheader("📦 匯出完整稽核報告")
        render_blob_download_button(
            st.session_state.zip_bytes,
            st.session_state.zip_filename,
            f"📥 點此直接下載【{mode_now}】RULA 報表與獨立擷取照片庫 (ZIP)"
        )

        # 4. Base64 內嵌影像預覽畫廊
        st.subheader("🖼️ 抽樣標註照片預覽（含五官網格、手掌拇指三角、足底支撐三角與全身雙側骨架）")
        if mode_now == "左右側同步計算":
            tab_img_pair, tab_img_r, tab_img_l = st.tabs([
                "👀 左右鏡頭同步對照檢視",
                "📸 右手側鏡頭獨立擷取照",
                "📸 左手側鏡頭獨立擷取照"
            ])
            with tab_img_pair:
                pair_count = min(len(st.session_state.display_right_b64), len(st.session_state.display_left_b64))
                for idx in range(pair_count):
                    c_r, c_l = st.columns(2)
                    cap_r, b64_r = st.session_state.display_right_b64[idx]
                    cap_l, b64_l = st.session_state.display_left_b64[idx]
                    with c_r:
                        st.markdown(
                            f"""
                            <div style="text-align:center; margin-bottom:16px; background:#1e1e1e; padding:8px; border-radius:8px;">
                                <img src="data:image/jpeg;base64,{b64_r}" style="width:100%; border-radius:6px;">
                                <div style="font-size:13px; color:#ddd; margin-top:6px;">{cap_r}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with c_l:
                        st.markdown(
                            f"""
                            <div style="text-align:center; margin-bottom:16px; background:#1e1e1e; padding:8px; border-radius:8px;">
                                <img src="data:image/jpeg;base64,{b64_l}" style="width:100%; border-radius:6px;">
                                <div style="font-size:13px; color:#ddd; margin-top:6px;">{cap_l}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            with tab_img_r:
                cols_r = st.columns(3)
                for idx, (cap_text, b64_str) in enumerate(st.session_state.display_right_b64):
                    with cols_r[idx % 3]:
                        st.markdown(
                            f"""
                            <div style="text-align:center; margin-bottom:16px; background:#1e1e1e; padding:8px; border-radius:8px;">
                                <img src="data:image/jpeg;base64,{b64_str}" style="width:100%; border-radius:6px;">
                                <div style="font-size:13px; color:#ddd; margin-top:6px;">{cap_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            with tab_img_l:
                cols_l = st.columns(3)
                for idx, (cap_text, b64_str) in enumerate(st.session_state.display_left_b64):
                    with cols_l[idx % 3]:
                        st.markdown(
                            f"""
                            <div style="text-align:center; margin-bottom:16px; background:#1e1e1e; padding:8px; border-radius:8px;">
                                <img src="data:image/jpeg;base64,{b64_str}" style="width:100%; border-radius:6px;">
                                <div style="font-size:13px; color:#ddd; margin-top:6px;">{cap_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            with st.expander("點此展開 / 收合所有抽樣標註照片", expanded=True):
                gallery_cols = st.columns(3)
                for idx, (cap_text, b64_str) in enumerate(st.session_state.display_images_b64):
                    with gallery_cols[idx % 3]:
                        st.markdown(
                            f"""
                            <div style="text-align:center; margin-bottom:16px; background:#1e1e1e; padding:8px; border-radius:8px;">
                                <img src="data:image/jpeg;base64,{b64_str}" style="width:100%; border-radius:6px;">
                                <div style="font-size:13px; color:#ddd; margin-top:6px;">{cap_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
    else:
        st.warning("⚠️ 影片中未偵測到有效的人體姿態關鍵點，請確認影片人物是否清晰完整。")
