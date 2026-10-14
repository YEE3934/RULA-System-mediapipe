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
import base64
import tempfile
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

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
    # 黑色描邊確保在任何背景皆清晰可見
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
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def calculate_3d_angle(a, b, c):
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-5 or norm_bc < 1e-5:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def enhance_image_for_pose(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def get_upper_arm_score(angle, is_abducted=False, is_raised=False):
    if angle <= 20: base = 1
    elif angle <= 45: base = 2
    elif angle <= 90: base = 3
    else: base = 4
    if is_abducted: base += 1
    if is_raised: base += 1
    return min(max(base, 1), 6)

def get_lower_arm_score(angle, is_across_midline=False):
    base = 1 if 60 <= angle <= 100 else 2
    if is_across_midline: base += 1
    return min(max(base, 1), 3)

def get_wrist_score(angle, is_deviated=False):
    # RULA Step 3: 0~5度=1分, 5~15度=2分, >15度=3分; 尺偏/橈偏 +1分
    if angle <= 5: base = 1
    elif angle <= 15: base = 2
    else: base = 3
    if is_deviated: base += 1
    return min(max(base, 1), 4)

def get_neck_score(angle, is_twisted=False, is_side_bending=False):
    if angle <= 10: base = 1
    elif angle <= 20: base = 2
    else: base = 3
    if is_twisted: base += 1
    if is_side_bending: base += 1
    return min(max(base, 1), 6)

def get_trunk_score(angle, is_twisted=False, is_side_bending=False):
    if angle <= 5: base = 1
    elif angle <= 20: base = 2
    elif angle <= 60: base = 3
    else: base = 4
    if is_twisted: base += 1
    if is_side_bending: base += 1
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

# ==================== 【全 33 點生物力學分層渲染器 (五官 + 手掌拇指三角 + 足底三角)】 ====================
def draw_full_biomechanical_skeleton(
    img, pts_dict, ang_ua, ang_la, ang_wr, ang_tk, ang_nk, wt_score
):
    """
    繪製完整 33 點解剖特徵：
    1. 臉部五官網格 (鼻、雙眼、雙耳、嘴角)
    2. 雙側完整骨架 (對側為半透明青色輔助層，主測側為高亮力學層)
    3. 手部精細結構 (手腕、拇指軸線、食指-小指手掌平面三角形)
    4. 足底支撐結構 (腳踝-腳跟-腳尖支撐三角形)
    """
    out = img.copy()
    overlay = img.copy()

    def to_pt(arr):
        return (int(round(arr[0])), int(round(arr[1])))

    # ---- 第一層：半透明色塊與對側輔助骨架 (繪製於 overlay) ----
    # 1. 軀幹核心網格與對側肢體
    o_sh, o_el, o_wr = to_pt(pts_dict["o_sh"]), to_pt(pts_dict["o_el"]), to_pt(pts_dict["o_wr"])
    o_th, o_ix, o_pk = to_pt(pts_dict["o_th"]), to_pt(pts_dict["o_ix"]), to_pt(pts_dict["o_pk"])
    o_hp, o_kn, o_an = to_pt(pts_dict["o_hp"]), to_pt(pts_dict["o_kn"]), to_pt(pts_dict["o_an"])
    o_hl, o_ft = to_pt(pts_dict["o_hl"]), to_pt(pts_dict["o_ft"])

    # 對側上肢與手掌 (青灰色輔助線)
    cv2.line(overlay, o_sh, o_el, (200, 180, 100), 2, cv2.LINE_AA)
    cv2.line(overlay, o_el, o_wr, (200, 180, 100), 2, cv2.LINE_AA)
    cv2.line(overlay, o_wr, o_th, (200, 220, 120), 2, cv2.LINE_AA)
    cv2.polylines(overlay, [np.array([o_wr, o_ix, o_pk], dtype=np.int32)], True, (200, 180, 100), 2, cv2.LINE_AA)

    # 對側下肢與足底三角
    cv2.line(overlay, o_hp, o_kn, (200, 160, 100), 2, cv2.LINE_AA)
    cv2.line(overlay, o_kn, o_an, (200, 160, 100), 2, cv2.LINE_AA)
    cv2.polylines(overlay, [np.array([o_an, o_hl, o_ft], dtype=np.int32)], True, (200, 160, 100), 2, cv2.LINE_AA)
    for p in [o_sh, o_el, o_wr, o_th, o_ix, o_pk, o_hp, o_kn, o_an, o_hl, o_ft]:
        cv2.circle(overlay, p, 4, (220, 200, 120), -1, cv2.LINE_AA)

    # 主評估側手掌三角形與足底三角形半透明填充
    p_wr = to_pt(pts_dict["wr"])
    p_th = to_pt(pts_dict["th"])
    p_ix = to_pt(pts_dict["ix"])
    p_pk = to_pt(pts_dict["pk"])
    p_an = to_pt(pts_dict["an"])
    p_hl = to_pt(pts_dict["hl"])
    p_ft = to_pt(pts_dict["ft"])

    palm_poly = np.array([p_wr, p_ix, p_pk], dtype=np.int32)
    foot_poly = np.array([p_an, p_hl, p_ft], dtype=np.int32)
    cv2.fillPoly(overlay, [palm_poly], (0, 220, 255))
    cv2.fillPoly(overlay, [foot_poly], (255, 200, 0))

    # 融合半透明圖層
    cv2.addWeighted(overlay, 0.42, out, 0.58, 0, out)

    # ---- 第二層：頭部五官解剖網格 (鼻、雙眼、雙耳、雙嘴角) ----
    p_nose = to_pt(pts_dict["nose"])
    p_leye = to_pt(pts_dict["l_eye"])
    p_reye = to_pt(pts_dict["r_eye"])
    p_lear = to_pt(pts_dict["l_ear"])
    p_rear = to_pt(pts_dict["r_ear"])
    p_lmou = to_pt(pts_dict["l_mouth"])
    p_rmou = to_pt(pts_dict["r_mouth"])
    p_ear  = to_pt(pts_dict["ear"])
    p_sh   = to_pt(pts_dict["sh"])
    p_el   = to_pt(pts_dict["el"])
    p_hp   = to_pt(pts_dict["hp"])
    p_kn   = to_pt(pts_dict["kn"])

    # 繪製臉部特徵連線 (耳 -> 眼 -> 鼻、雙眼連線、嘴角連線、鼻至嘴中線)
    cv2.line(out, p_lear, p_leye, (255, 255, 100), 2, cv2.LINE_AA)
    cv2.line(out, p_rear, p_reye, (255, 255, 100), 2, cv2.LINE_AA)
    cv2.line(out, p_leye, p_nose, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(out, p_reye, p_nose, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(out, p_leye, p_reye, (255, 220, 0), 2, cv2.LINE_AA)
    cv2.line(out, p_lmou, p_rmou, (180, 100, 255), 2, cv2.LINE_AA)
    mouth_mid = ((p_lmou[0] + p_rmou[0]) // 2, (p_lmou[1] + p_rmou[1]) // 2)
    cv2.line(out, p_nose, mouth_mid, (0, 255, 255), 2, cv2.LINE_AA)

    for fp in [p_nose, p_leye, p_reye, p_lear, p_rear, p_lmou, p_rmou]:
        cv2.circle(out, fp, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, fp, 3, (0, 180, 255), -1, cv2.LINE_AA)

    # ---- 第三層：主評估側核心人因力學鏈 ----
    # 垂直參考基準線 (過髖關節正上方)
    cv2.line(out, p_hp, (p_hp[0], max(0, p_sh[1] - 70)), (190, 190, 190), 2, cv2.LINE_AA)

    # 1. 下肢與足底支撐三角形 (髖 -> 膝 -> 踝 -> 跟/尖)
    cv2.line(out, p_hp, p_kn, (255, 170, 0), 4, cv2.LINE_AA)
    cv2.line(out, p_kn, p_an, (255, 170, 0), 4, cv2.LINE_AA)
    cv2.polylines(out, [foot_poly], True, (255, 220, 0), 3, cv2.LINE_AA)

    # 2. 軀幹與頭頸軸線 (髖 -> 肩 -> 耳 -> 鼻)
    cv2.line(out, p_hp, p_sh, (0, 230, 0), 5, cv2.LINE_AA)
    cv2.line(out, p_sh, p_ear, (0, 255, 255), 4, cv2.LINE_AA)
    cv2.line(out, p_ear, p_nose, (0, 215, 255), 3, cv2.LINE_AA)

    # 3. 上肢與手掌力學三角 (肩 -> 肘 -> 腕 + 拇指向量 + 食指/小指掌面三角)
    cv2.line(out, p_sh, p_el, (0, 70, 255), 5, cv2.LINE_AA)
    cv2.line(out, p_el, p_wr, (0, 150, 255), 5, cv2.LINE_AA)
    # 手掌平面三角形 (腕 - 食指 - 小指)
    cv2.polylines(out, [palm_poly], True, (0, 255, 255), 3, cv2.LINE_AA)
    # 拇指獨立向量 (粉紫色高亮，用於呈現手腕翻轉方向)
    cv2.line(out, p_wr, p_th, (255, 50, 255), 3, cv2.LINE_AA)

    # 繪製主側關鍵點節點
    main_joints = [p_ear, p_sh, p_el, p_wr, p_hp, p_kn, p_an]
    for pt in main_joints:
        cv2.circle(out, pt, 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)

    # 繪製手部與足部端點節點 (拇指、食指、小指、腳跟、腳尖)
    extremity_joints = [p_th, p_ix, p_pk, p_hl, p_ft]
    for pt in extremity_joints:
        cv2.circle(out, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, pt, 4, (255, 0, 200), -1, cv2.LINE_AA)

    # 標註各關節即時角度與手腕扭轉狀態
    twist_str = "極端扭轉(2分)" if wt_score == 2 else "中立(1分)"
    out = put_chinese_text(out, f"頸部:{ang_nk:.0f}°", (p_ear[0] + 12, max(10, p_ear[1] - 12)), (255, 255, 120), 14)
    out = put_chinese_text(out, f"上臂:{ang_ua:.0f}°", (p_sh[0] + 12, p_sh[1] - 8), (0, 255, 255), 14)
    out = put_chinese_text(out, f"前臂:{ang_la:.0f}°", (p_el[0] + 12, p_el[1] + 2), (0, 200, 255), 14)
    out = put_chinese_text(out, f"手腕:{ang_wr:.0f}°|{twist_str}", (p_wr[0] + 12, p_wr[1] + 5), (255, 160, 255), 14)
    out = put_chinese_text(out, f"軀幹:{ang_tk:.0f}°", (p_hp[0] + 12, p_hp[1] - 15), (100, 255, 100), 14)
    return out

# ==================== 【正面鏡頭：冠狀面/水平面補正分析】 ====================
def analyze_front_camera(frame, pose_model, target_side):
    orig_h, orig_w = frame.shape[:2]
    scale = min(800 / max(orig_h, orig_w), 1.0)
    frame_resized = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
    
    img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    results = pose_model.process(img_rgb)
    
    if not results.pose_landmarks:
        return False, {}
        
    lm = results.pose_landmarks.landmark
    
    sh_idx = 12 if target_side == "右手側" else 11
    el_idx = 14 if target_side == "右手側" else 13
    wr_idx = 16 if target_side == "右手側" else 15
    
    midline_x = (lm[11].x + lm[12].x + lm[23].x + lm[24].x) / 4.0
    
    shoulder_width = abs(lm[11].x - lm[12].x)
    elbow_out_dist = (lm[el_idx].x - lm[sh_idx].x) if target_side == "右手側" else (lm[sh_idx].x - lm[el_idx].x)
    is_abducted = elbow_out_dist > shoulder_width * 0.35
    
    is_across_midline = (lm[wr_idx].x < midline_x) if target_side == "右手側" else (lm[wr_idx].x > midline_x)
    
    shoulder_tilt = abs(lm[11].y - lm[12].y)
    is_trunk_side_bending = shoulder_tilt > 0.08

    # 正面視角檢測頭部/頸部是否明顯側轉或側傾
    ear_diff_y = abs(lm[7].y - lm[8].y)
    is_neck_side_bending = ear_diff_y > shoulder_width * 0.18
    
    mods = {
        "is_abducted": is_abducted,
        "is_across_midline": is_across_midline,
        "is_side_bending": is_trunk_side_bending,
        "is_neck_side_bending": is_neck_side_bending
    }
    return True, mods

# ==================== 【側視角鏡頭：33 點全解剖擷取與 RULA 全矩陣解算】 ====================
def analyze_side_camera(frame, pose_model, target_side, front_mods, m_val, f_val):
    orig_h, orig_w = frame.shape[:2]
    scale = min(1000 / max(orig_h, orig_w), 1.0)
    work_w, work_h = int(orig_w * scale), int(orig_h * scale)
    frame_work = cv2.resize(frame, (work_w, work_h))
    
    enhanced_bgr = enhance_image_for_pose(frame_work)
    img_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    results = pose_model.process(img_rgb)
    
    if not results.pose_landmarks or not results.pose_world_landmarks:
        return False, None, frame_work, 0.0
        
    n_lm = results.pose_landmarks.landmark
    w_lm = results.pose_world_landmarks.landmark
    
    try:
        def pt2d(idx):
            return np.array([n_lm[idx].x * work_w, n_lm[idx].y * work_h], dtype=float)

        def pt3d(idx):
            return np.array([w_lm[idx].x, w_lm[idx].y, w_lm[idx].z], dtype=float)

        # 定義主評估側與對側之 33 點索引 (含五官、手腕三指、足部三點)
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

        # 計算綜合信賴度 (納入肩、肘、腕、拇指、食指、髖、膝、踝)
        key_vis = [
            n_lm[t_sh].visibility, n_lm[t_el].visibility, n_lm[t_wr].visibility,
            n_lm[t_th].visibility, n_lm[t_ix].visibility, n_lm[t_hp].visibility,
            n_lm[t_kn].visibility, n_lm[t_an].visibility
        ]
        conf_side = float(np.mean(key_vis) * 100)

        # 1. 軀幹軸心穩定與臉部五官提取
        sh = pt2d(t_sh) * 0.65 + pt2d(o_sh) * 0.35
        hp = pt2d(t_hp) * 0.65 + pt2d(o_hp) * 0.35
        ear = pt2d(t_ear) if n_lm[t_ear].visibility >= n_lm[o_ear].visibility else pt2d(o_ear)
        nose = pt2d(0)
        l_eye, r_eye = pt2d(2), pt2d(5)
        l_ear_pt, r_ear_pt = pt2d(7), pt2d(8)
        l_mouth, r_mouth = pt2d(9), pt2d(10)

        # 2. 主側與對側之上肢 + 手掌三指提取 (防雙手重疊反轉)
        el_t, wr_t, th_t, ix_t, pk_t = pt2d(t_el), pt2d(t_wr), pt2d(t_th), pt2d(t_ix), pt2d(t_pk)
        el_o, wr_o, th_o, ix_o, pk_o = pt2d(o_el), pt2d(o_wr), pt2d(o_th), pt2d(o_ix), pt2d(o_pk)

        dist_t = np.linalg.norm(wr_t - sh)
        dist_o = np.linalg.norm(wr_o - sh)
        if n_lm[o_wr].visibility > n_lm[t_wr].visibility + 0.15 or (dist_o > dist_t * 1.25 and n_lm[o_wr].visibility > 0.4):
            el, wr, th, ix, pk = el_o, wr_o, th_o, ix_o, pk_o
            b_el, b_wr, b_th, b_ix, b_pk = el_t, wr_t, th_t, ix_t, pk_t
            w3_el, w3_wr, w3_th, w3_ix, w3_pk = pt3d(o_el), pt3d(o_wr), pt3d(o_th), pt3d(o_ix), pt3d(o_pk)
            conf_side = max(conf_side, float(n_lm[o_wr].visibility * 100))
        else:
            el, wr, th, ix, pk = el_t, wr_t, th_t, ix_t, pk_t
            b_el, b_wr, b_th, b_ix, b_pk = el_o, wr_o, th_o, ix_o, pk_o
            w3_el, w3_wr, w3_th, w3_ix, w3_pk = pt3d(t_el), pt3d(t_wr), pt3d(t_th), pt3d(t_ix), pt3d(t_pk)

        # 解剖學上臂長度約束 (避免寬大黑色袖口把肘關節往下拉)
        torso_len = np.linalg.norm(hp - sh)
        upper_len = np.linalg.norm(el - sh)
        if torso_len > 10 and upper_len > torso_len * 0.72:
            el = sh + (el - sh) / upper_len * (torso_len * 0.62)

        # 3. 下肢與足底三角形提取 (踝、跟、尖)
        kn, an, hl, ft = pt2d(t_kn), pt2d(t_an), pt2d(t_hl), pt2d(t_ft)
        b_kn, b_an, b_hl, b_ft = pt2d(o_kn), pt2d(o_an), pt2d(o_hl), pt2d(o_ft)

        # ==================== 【RULA 全關節幾何角度解算】 ====================
        # 1. 前臂屈曲角 (Lower Arm): 180 - 肘關節內角
        ang_lower_arm = abs(180.0 - calculate_2d_angle(sh, el, wr))
        
        # 2. 上臂屈曲角 (Upper Arm): 相對於軀幹主軸向量 (肩 -> 髖)
        trunk_vec = hp - sh
        ref_sh_down = sh + trunk_vec
        ang_upper_arm = calculate_2d_angle(ref_sh_down, sh, el)
        
        # 3. 手腕屈伸角 (Wrist Flexion/Extension - RULA Step 3):
        # 利用手掌中心 (食指與小指中點) 與手腕連線作為「手掌軸線」，計算與前臂延伸線之夾角
        palm_mid_2d = (ix + pk) * 0.5
        palm_len_2d = np.linalg.norm(palm_mid_2d - wr)
        if palm_len_2d > 6.0:
            raw_wrist_2d = abs(180.0 - calculate_2d_angle(el, wr, palm_mid_2d))
        else:
            raw_wrist_2d = 0.0
        # 結合 3D 手腕角度抑制側視角透視折疊誤差，並限制在合理生理範圍 (0~60度)
        palm_mid_3d = (w3_ix + w3_pk) * 0.5
        raw_wrist_3d = abs(180.0 - calculate_3d_angle(w3_el, w3_wr, palm_mid_3d))
        ang_wrist = float(np.clip(raw_wrist_2d * 0.6 + raw_wrist_3d * 0.4, 0.0, 60.0))

        # 4. 手腕扭轉與尺橈偏判定 (Wrist Twist & Deviation - RULA Step 4):
        # 透過拇指 (Thumb) 與小指 (Pinky) 的掌寬向量與前臂軸線關係判定旋前/旋後扭轉
        forearm_vec_3d = w3_wr - w3_el
        palm_span_3d = w3_th - w3_pk
        # 計算掌寬向量與垂直軸(Y)及前臂之空間翻轉關係
        twist_angle = calculate_3d_angle(w3_el, w3_wr, w3_th)
        is_wrist_deviated = abs(180.0 - twist_angle) > 25.0
        # 若拇指在 2D 視角明顯低於小指或手掌平面大幅度翻轉，判定為極端扭轉 (2分)，否則為中立 (1分)
        thumb_pinky_dy = (th[1] - pk[1]) / max(np.linalg.norm(th - pk), 1.0)
        wt_score = 2 if abs(thumb_pinky_dy) > 0.65 else 1

        # 5. 軀幹前傾角 (Trunk - RULA Step 10): 髖至肩連線與垂直線夾角
        vertical_up = np.array([hp[0], hp[1] - 100.0])
        ang_trunk = calculate_2d_angle(vertical_up, hp, sh)
        
        # 6. 頭頸部屈曲與扭轉 (Neck - RULA Step 9):
        # 綜合耳肩連線與臉部朝向 (耳->鼻/眼)，反應真實低頭或抬頭視物姿勢
        head_center = ear * 0.6 + ((l_eye + r_eye) * 0.5) * 0.4
        ang_neck = abs(180.0 - calculate_2d_angle(hp, sh, head_center))

        # 7. 下肢足底支撐平衡判定 (Legs - RULA Step 11):
        # 比較雙腳腳跟與腳尖接地高度差，若單腳明顯懸空 (>軀幹長度 20%) 則記為 2 分(不穩)，平穩為 1 分
        foot_height_diff = abs( max(hl[1], ft[1]) - max(b_hl[1], b_ft[1]) )
        lg_score = 2 if (torso_len > 10 and foot_height_diff > torso_len * 0.22) else 1

        # ==================== 【RULA 查表綜合評分】 ====================
        s_ua = get_upper_arm_score(ang_upper_arm, is_abducted=front_mods.get("is_abducted", False))
        s_la = get_lower_arm_score(ang_lower_arm, is_across_midline=front_mods.get("is_across_midline", False))
        s_wr = get_wrist_score(ang_wrist, is_deviated=is_wrist_deviated)
        s_tk = get_trunk_score(ang_trunk, is_side_bending=front_mods.get("is_side_bending", False))
        s_nk = get_neck_score(ang_neck, is_side_bending=front_mods.get("is_neck_side_bending", False))
        
        sc_a, sc_b, sc_c, sc_d, grand, al = compute_rula_full(
            s_ua, s_la, s_wr, wt_score, s_nk, s_tk, lg_score, m_val, f_val
        )
        
        # 打包所有 33 點解剖座標供繪圖器渲染
        pts_dict = {
            "nose": nose, "l_eye": l_eye, "r_eye": r_eye,
            "l_ear": l_ear_pt, "r_ear": r_ear_pt, "l_mouth": l_mouth, "r_mouth": r_mouth,
            "ear": ear, "sh": sh, "el": el, "wr": wr, "th": th, "ix": ix, "pk": pk,
            "hp": hp, "kn": kn, "an": an, "hl": hl, "ft": ft,
            "o_sh": pt2d(o_sh), "o_el": b_el, "o_wr": b_wr, "o_th": b_th, "o_ix": b_ix, "o_pk": b_pk,
            "o_hp": pt2d(o_hp), "o_kn": b_kn, "o_an": b_an, "o_hl": b_hl, "o_ft": b_ft
        }

        frame_annotated = draw_full_biomechanical_skeleton(
            frame_work, pts_dict,
            ang_upper_arm, ang_lower_arm, ang_wrist, ang_trunk, ang_neck, wt_score
        )
        
        side_prefix = "右手" if target_side == "右手側" else "左手"
        data = {
            "頸部角度": round(ang_neck, 1),
            "軀幹角度": round(ang_trunk, 1),
            f"{side_prefix}上臂角度": round(ang_upper_arm, 1),
            f"{side_prefix}前臂角度": round(ang_lower_arm, 1),
            f"{side_prefix}手腕角度": round(ang_wrist, 1),
            f"{side_prefix}手腕扭轉": "極端扭轉(2分)" if wt_score == 2 else "中立(1分)",
            "足底支撐狀態": "不平衡(2分)" if lg_score == 2 else "雙足平穩(1分)",
            f"{side_prefix}上臂外展補正": "是 (+1分)" if front_mods.get("is_abducted") else "否",
            f"{side_prefix}橫越中線補正": "是 (+1分)" if front_mods.get("is_across_midline") else "否",
            "軀幹側彎補正": "是 (+1分)" if front_mods.get("is_side_bending") else "否",
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
        return False, None, frame_work, 0.0


# ==================== 【Streamlit 網頁應用介面】 ====================
st.set_page_config(page_title="RULA 三視角人因工程 AI 評估系統", layout="wide")
st.title("RULA 姿勢危害評估系統")
st.markdown(
    "> **全 33 點生物力學架構說明**：系統完整擷取**臉部五官（鼻/眼/耳/嘴）**、**手掌平面與拇指軸線（腕/拇指/食指/小指）**及**足底支撐三角（踝/腳跟/腳尖）**，同步解算上臂、前臂、手腕屈伸、手腕扭轉、頭頸、軀幹與下肢支撐之完整 RULA 矩陣。"
)

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.df_records = None
    st.session_state.display_images_b64 = []
    st.session_state.zip_bytes = None
    st.session_state.zip_filename = "RULA_Report.zip"
    st.session_state.analyzed_side = ""

target_side = st.radio(
    "請選擇本次作業分析之【主要評估側】:",
    options=["右手側", "左手側"],
    index=0,
    horizontal=True,
    help="主評估側將以高亮力學色塊標示（含手掌三角形與拇指扭轉軸），對側則以半透明輔助骨架呈現。"
)

st.header("第一步：上傳視角影片")
col_front, col_right, col_left = st.columns(3)

with col_front:
    vid_front_file = st.file_uploader("1. 正面鏡頭 (必填: 判定外展/交叉/側傾)", type=['mp4', 'mov', 'avi'])
with col_right:
    vid_right_file = st.file_uploader("2. 右側鏡頭 (量測右側動作)", type=['mp4', 'mov', 'avi'])
with col_left:
    vid_left_file = st.file_uploader("3. 左側鏡頭 (量測左側動作)", type=['mp4', 'mov', 'avi'])

st.header("第二步：等距抽樣與負載參數設定")
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    num_samples = st.slider("等距抽樣張數", min_value=5, max_value=100, value=25)
with col_p2:
    combo_muscle = st.selectbox("肌肉使用分數 (Muscle Score):", ["0: 靜態少於1分鐘 / 偶發動作", "1: 姿勢維持>1分鐘 / 高重複作業"])
with col_p3:
    combo_force = st.selectbox("負載與施力 (Force/Load):", ["0: < 2kg (間歇施力)", "1: 2-10kg (間歇施力)", "2: 2-10kg (靜態/重複施力)", "3: > 10kg 或快速衝擊力"])

# 執行分析按鈕
if st.button("🚀 啟動全關節生物力學分析", type="primary", use_container_width=True):
    selected_side_vid = vid_right_file if target_side == "右手側" else vid_left_file
    
    if not vid_front_file:
        st.error("❌ 缺少【正面鏡頭影片】！正面鏡頭是用於校正水平面外展與中線穿越的必要基準。")
    elif not selected_side_vid:
        st.error(f"❌ 缺少【{target_side}側面影片】！您選擇評估「{target_side}」，必須提供該側面的拍攝視角進行主屈曲量測。")
    else:
        with st.spinner(f"系統正在進行 33 點全解剖特徵擷取與三視角聯立解算（目標：{target_side}）..."):
            st.session_state.display_images_b64 = []
            st.session_state.zip_bytes = None
            gc.collect()

            t_front = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_front.write(vid_front_file.getvalue())
            t_front.close()
            
            t_side = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            t_side.write(selected_side_vid.getvalue())
            t_side.close()

            cap_front = cv2.VideoCapture(t_front.name)
            cap_side = cv2.VideoCapture(t_side.name)

            fps = cap_side.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = min(int(cap_front.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_side.get(cv2.CAP_PROP_FRAME_COUNT)))

            if total_frames <= 0:
                cap_front.release()
                cap_side.release()
                os.remove(t_front.name)
                os.remove(t_side.name)
                st.error("影片解析長度為 0，請檢查影片編碼格式！")
                st.stop()

            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()

            m_val = int(combo_muscle.split(":")[0])
            f_val = int(combo_force.split(":")[0])

            records = []
            zip_images = []
            display_images_b64 = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            with mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.55
            ) as pose:
                total_samples = len(sample_indices)
                for i, idx in enumerate(sample_indices):
                    sec = round(idx / fps, 2)
                    cap_front.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    cap_side.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    s_f, f_frame = cap_front.read()
                    s_s, s_frame = cap_side.read()

                    if s_f and s_s:
                        _, front_mods = analyze_front_camera(f_frame, pose, target_side)
                        succ_s, data, side_annotated, conf = analyze_side_camera(s_frame, pose, target_side, front_mods, m_val, f_val)

                        if succ_s and data is not None:
                            record = {
                                "樣本編號": i + 1,
                                "時間(秒)": sec,
                                "主要評估側": target_side,
                                "視角信賴度(%)": round(conf, 1)
                            }
                            record.update(data)
                            records.append(record)

                            # 左上角資訊看板標註
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

                    progress_bar.progress((i + 1) / total_samples)
                    status_text.text(f"全關節姿態解算進度: {i+1} / {total_samples} 筆完成")

            cap_front.release()
            cap_side.release()
            try:
                os.remove(t_front.name)
                os.remove(t_side.name)
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
            st.session_state.analyzed_side = target_side
            gc.collect()

# ==================== 【網頁即時成果展示與打包下載區】 ====================
if st.session_state.analysis_done:
    st.divider()
    st.success(f"✅ 【{st.session_state.analyzed_side}】全 33 點多視角生物力學解算完成！以下為即時分析結果：")
    
    df_show = st.session_state.df_records
    
    if df_show is not None and not df_show.empty:
        # 1. 核心人因指標看板
        st.subheader("📈 RULA 危害指標總覽")
        avg_score = df_show["最終最高分"].mean()
        max_score = df_show["最終最高分"].max()
        high_risk_pct = (df_show["最終最高分"] >= 5).mean() * 100
        avg_conf = df_show["視角信賴度(%)"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("平均 RULA 總分", f"{avg_score:.2f} 分")
        m2.metric("最高風險峰值", f"{max_score} 分")
        m3.metric("高危害佔比 (AL3/AL4)", f"{high_risk_pct:.1f}%")
        m4.metric("平均視角信賴度", f"{avg_conf:.1f}%")

        # 2. 互動式數據報表
        st.subheader("📊 詳細抽樣數據報表（含手腕屈伸、手腕扭轉與足底支撐）")
        st.dataframe(df_show, use_container_width=True)

        # 3. 瀏覽器本地直出 ZIP 下載按鈕
        st.subheader("📦 匯出完整稽核報告")
        render_blob_download_button(
            st.session_state.zip_bytes,
            st.session_state.zip_filename,
            f"📥 點此直接下載 {st.session_state.analyzed_side} RULA 分析報告與照片庫 (ZIP)"
        )

        # 4. Base64 內嵌影像預覽畫廊
        st.subheader("🖼️ 抽樣標註照片預覽（含五官網格、手掌拇指三角、足底支撐三角）")
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
