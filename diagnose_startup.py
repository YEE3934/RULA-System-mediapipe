import streamlit as st
import sys, platform, time
st.set_page_config(page_title="RULA 部署診斷", layout="centered")
st.title("RULA 部署診斷")
st.success("第一關通過：Streamlit 網頁可以正常載入。")
st.write("Python:", sys.version.split()[0])
st.write("Streamlit:", st.__version__)
st.write("作業系統:", platform.platform())
st.info("下面的按鈕會在你按下後才載入較重的套件，不影響首頁啟動。")
if st.button("測試 OpenCV / NumPy / MediaPipe 載入"):
    with st.spinner("正在逐項載入..."):
        try:
            import numpy as np
            st.write("NumPy:", np.__version__)
            import cv2
            st.write("OpenCV:", cv2.__version__)
            import mediapipe as mp
            st.write("MediaPipe:", mp.__version__)
            st.write("Legacy Pose API:", hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"))
            st.success("第二關通過：核心套件匯入正常。")
        except Exception as e:
            st.exception(e)
