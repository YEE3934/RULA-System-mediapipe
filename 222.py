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
APP_BUILD_TAG = "RULA_AUDITABLE_2026-10-24_v3"

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
