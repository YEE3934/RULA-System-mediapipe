import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import math
import numpy as np
import pandas as pd
import os
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

# ==================== 【繞過 /media/ 的瀏覽器直出下載模組】 ====================
def render_blob_download_button(zip_bytes, filename, label_text):
    """使用 JS Blob 直接在瀏覽器記憶體合成檔案並觸發下載，徹底解決無法存取網站問題"""
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

# ==================== 【人因工程規範幾何計算】 ====================
def calculate_3d_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: 
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

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

def compute_rula_side_score(ua, la, nk, tk, m_val, f_val):
    w, wt, lg = 1, 1, 1
    score_a = TABLE_A[ua][la][w][wt]
    score_b = TABLE_B[nk][tk][lg]
    score_c = score_a + m_val + f_val
    score_d = score_b + m_val + f_val
    grand_score = TABLE_C[min(score_c, 8) - 1][min(score_d, 7) - 1]
    al_num = 1 if grand_score <= 2 else (2 if grand_score <= 4 else (3 if grand_score <= 6 else 4))
    return score_a, score_b, score_c, score_d, grand_score, f"AL{al_num}"

def put_chinese_text(img, text, position, color, size=15):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font = None
    font_candidates = [
        "font.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "msjh.ttc",
        "simhei.ttf"
    ]
    for p in font_candidates:
        if os.path.exists(p):
            try:
                font = ImageFont.truetype(p, size)
                break
            except Exception:
                continue
                
    if font is None:
        font = ImageFont.load_default()
        
    draw.text(position, text, fill=(color[2], color[1], color[0]), font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ==================== 【正面鏡頭：冠狀面/水平面補正分析】 ====================
def analyze_front_camera(frame, pose_model, target_side):
    orig_h, orig_w = frame.shape[:2]
    scale = min(640 / orig_w, 480 / orig_h)
    frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    
    mods = {
        "is_abducted": is_abducted,
        "is_across_midline": is_across_midline,
        "is_side_bending": is_trunk_side_bending
    }
    return True, mods

# ==================== 【側視角鏡頭：矢狀面主要人因量測】 ====================
def analyze_side_camera(frame, pose_model, target_side, front_mods, m_val, f_val):
    orig_h, orig_w = frame.shape[:2]
    scale = min(640 / orig_w, 480 / orig_h)
    frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(img_rgb)
    frame_annotated = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    if not results.pose_world_landmarks or not results.pose_landmarks:
        return False, None, frame_annotated, 0.0
        
    mp.solutions.drawing_utils.draw_landmarks(frame_annotated, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    w_lm = results.pose_world_landmarks.landmark
    n_lm = results.pose_landmarks.landmark
    
    try:
        sh_idx = 12 if target_side == "右手側" else 11
        el_idx = 14 if target_side == "右手側" else 13
        wr_idx = 16 if target_side == "右手側" else 15
        hip_idx = 24 if target_side == "右手側" else 23
        ear_idx = 8 if target_side == "右手側" else 7
        
        visibilities = [n_lm[sh_idx].visibility, n_lm[el_idx].visibility, n_lm[wr_idx].visibility, n_lm[hip_idx].visibility]
        conf_side = float(np.mean(visibilities) * 100)
        
        sh = [w_lm[sh_idx].x, w_lm[sh_idx].y, w_lm[sh_idx].z]
        el = [w_lm[el_idx].x, w_lm[el_idx].y, w_lm[el_idx].z]
        wr = [w_lm[wr_idx].x, w_lm[wr_idx].y, w_lm[wr_idx].z]
        hp = [w_lm[hip_idx].x, w_lm[hip_idx].y, w_lm[hip_idx].z]
        ear = [w_lm[ear_idx].x, w_lm[ear_idx].y, w_lm[ear_idx].z]
        
        ang_lower_arm = abs(180.0 - calculate_3d_angle(sh, el, wr))
        
        trunk_vec = [hp[0] - sh[0], hp[1] - sh[1], hp[2] - sh[2]]
        ref_sh_down = [sh[0] + trunk_vec[0], sh[1] + trunk_vec[1], sh[2] + trunk_vec[2]]
        ang_upper_arm = calculate_3d_angle(ref_sh_down, sh, el)
        
        vertical_up = [hp[0], hp[1] - 1.0, hp[2]]
        ang_trunk = calculate_3d_angle(vertical_up, hp, sh)
        
        ang_neck = abs(180.0 - calculate_3d_angle(hp, sh, ear))
        
        s_ua = get_upper_arm_score(ang_upper_arm, is_abducted=front_mods.get("is_abducted", False))
        s_la = get_lower_arm_score(ang_lower_arm, is_across_midline=front_mods.get("is_across_midline", False))
        s_tk = get_trunk_score(ang_trunk, is_side_bending=front_mods.get("is_side_bending", False))
        s_nk = get_neck_score(ang_neck)
        
        sc_a, sc_b, sc_c, sc_d, grand, al = compute_rula_side_score(s_ua, s_la, s_nk, s_tk, m_val, f_val)
        
        side_prefix = "右手" if target_side == "右手側" else "左手"
        data = {
            "頸部角度": round(ang_neck, 1),
            "軀幹角度": round(ang_trunk, 1),
            f"{side_prefix}上臂角度": round(ang_upper_arm, 1),
            f"{side_prefix}前臂角度": round(ang_lower_arm, 1),
            f"{side_prefix}上臂外展補正": "是 (+1分)" if front_mods.get("is_abducted") else "否",
            f"{side_prefix}橫越中線補正": "是 (+1分)" if front_mods.get("is_across_midline") else "否",
            "軀幹側彎補正": "是 (+1分)" if front_mods.get("is_side_bending") else "否",
            f"{side_prefix} Score A": sc_a,
            f"{side_prefix} Score C": sc_c,
            f"{side_prefix} Grand Score": grand,
            f"{side_prefix} AL": al,
            "最危害側": target_side,
            "最終最高分": grand,
            "最終 AL": al
        }
        return True, data, frame_annotated, conf_side
    except Exception:
        return False, None, frame_annotated, 0.0


# ==================== 【Streamlit 網頁應用介面】 ====================
st.set_page_config(page_title="RULA 三視角人因工程 AI 評估系統", layout="wide")
st.title("RULA 姿勢危害評估系統")
st.markdown("> **人因工程架構說明**：以側視角量測主要關節屈曲角度；正面視角判定上臂外展、橫越中線與側彎等額外危害扣分項。")

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
    help="根據 RULA 規範，評估員應指定作業負擔較重、受力較大或主要操作之單側手部進行評級。"
)

st.header("第一步：上傳視角影片")
col_front, col_right, col_left = st.columns(3)

with col_front:
    vid_front_file = st.file_uploader("1. 正面鏡頭 (必填: 判定外展/交叉)", type=['mp4', 'mov', 'avi'])
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
if st.button("🚀 啟動分析", type="primary", use_container_width=True):
    selected_side_vid = vid_right_file if target_side == "右手側" else vid_left_file
    
    if not vid_front_file:
        st.error("❌ 缺少【正面鏡頭影片】！正面鏡頭是用於校正水平面外展與中線穿越的必要基準。")
    elif not selected_side_vid:
        st.error(f"❌ 缺少【{target_side}側面影片】！您選擇評估「{target_side}」，必須提供該側面的拍攝視角進行主屈曲量測。")
    else:
        with st.spinner(f"系統正在進行三視角聯立解算（目標：{target_side}）..."):
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

            with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.6) as pose:
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

                            side_annotated = put_chinese_text(side_annotated, f"時間: {sec}s | 目標: {target_side}", (12, 12), (0, 255, 255), 16)
                            side_annotated = put_chinese_text(side_annotated, f"Grand Score: {data['最終最高分']} 分 ({data['最終 AL']})", (12, 38), (0, 0, 255) if data['最終最高分'] > 4 else (0, 255, 0), 15)
                            side_annotated = put_chinese_text(side_annotated, f"外展:{data[f'{target_side[:2]}上臂外展補正']} | 交叉:{data[f'{target_side[:2]}橫越中線補正']}", (12, 64), (255, 150, 0), 13)

                            is_ok, buf = cv2.imencode(".jpg", side_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                            if is_ok:
                                jpg_bytes = buf.tobytes()
                                side_en = "Right" if target_side == "右手側" else "Left"
                                zip_images.append((f"Sample_{i+1:03d}_{sec}s_{side_en}.jpg", jpg_bytes))
                                
                                # 轉成 Base64 字串直接嵌進 HTML，不走 /media/ 通道
                                b64_img = base64.b64encode(jpg_bytes).decode("utf-8")
                                caption_str = f"樣本 #{i+1} ({sec}s) | 總分: {data['最終最高分']} ({data['最終 AL']})"
                                display_images_b64.append((caption_str, b64_img))

                    progress_bar.progress((i + 1) / total_samples)
                    status_text.text(f"多視角姿態解算進度: {i+1} / {total_samples} 筆完成")

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
    st.success(f"✅ 【{st.session_state.analyzed_side}】多視角解算完成！以下為即時分析結果：")
    
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
        st.subheader("📊 詳細抽樣數據報表")
        st.dataframe(df_show, use_container_width=True)

        # 3. 瀏覽器本地直出 ZIP 下載按鈕 (繞過 Streamlit 伺服器 /media/ 限制)
        st.subheader("📦 匯出完整稽核報告")
        render_blob_download_button(
            st.session_state.zip_bytes,
            st.session_state.zip_filename,
            f"📥 點此直接下載 {st.session_state.analyzed_side} RULA 分析報告與照片庫 (ZIP)"
        )

        # 4. Base64 內嵌影像預覽畫廊 (不透過 /media/，100% 保證顯示不破圖)
        st.subheader("🖼️ 抽樣標註照片預覽")
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
