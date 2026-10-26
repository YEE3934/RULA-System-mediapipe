# RULA v8｜逐步導覽＋被剔除影格複核＋頸部候選分析

## 更新方式

1. 建議先保留 v7.2 GitHub 備份或建立測試分支。
2. 將此 ZIP 根目錄的 **`222.py`**、`requirements.txt`、`packages.txt` 上傳至 Streamlit Cloud 專案所使用的 GitHub 儲存庫**根目錄**。
3. 如已有 `models/hand_landmarker.task`，請繼續保留；沒有的話依 `models/README.txt` 下載官方 `.task` 模型，放入 `models/`。程式另會嘗試下載至可寫入的暫存路徑，但部署端網路可能不可用。
4. Streamlit Community Cloud 請選 **Python 3.11**；Main file path 為 `222.py`。
5. 網站頂部應出現 `RULA_V8_WIZARD_REJECT_NECK_2026-10-27`。若仍看到 v7.x，請先確認 GitHub 分支與 Cloud 部署路徑。

## 新版操作方式

頁面頂部與底部均提供「上一步／下一步」，左側有 **快速跳轉選單**。同一時間只顯示一個主要步驟；設定值存於目前 Streamlit Session State，不會因切換步驟而重置。**重新關閉網站或 Session 失效後，仍需重新上傳影片**（此版本並非永久儲存服務）。

1. 影片與基本設定
2. 自動辨識設定（各條件可選擇自動或人工）
3. 人工條件（腿足支撐、手腕扭轉、受力相關條件）
4. Hand 實驗設定（預設安全對照模式）
5. 啟動分析
6. 風險總覽
7. 被剔除樣本複核
8. 頸部品質比較
9. 手腕演算法比較
10. 異常與人工修正（含風險時間軸）
11. 影格照片
12. 完整報表與 ZIP 下載

## v8 新功能與可驗收項目

### 1. 被剔除樣本複核

- 只顯示自動分析跳過的**樣本＋側別**，而不是只記錄失敗數量。
- 每筆可以選擇正面、右側或左側**實際有影格的鏡頭**；左圖為原始未渲染 JPEG；右圖為同一影格再次執行 MediaPipe 後，**僅畫出可見度達到門檻的實際關鍵點**。完全未偵測到則不憑空繪製骨架。
- 可以標記「尚未複核」、「需要人工量測」、「照片可辨識但不恢復自動評分」、「確認無法辨識」。標記不會改變原始 RULA 分數。
- `Diagnostics/Rejected_Samples_Detail.csv` 包含樣本編號／時間／側別／原因及圖片相對路徑；`Diagnostics/Rejected_Images/` 保存原圖與診斷圖；下載時另存 `Diagnostics/Rejected_Manual_Reviews.csv`。
- **注意：右圖是事後使用相同影格重跑模型的診斷嘗試，可能不完全等於原始分析當下的內部結果**；因此保留拒絕原因，不因第二次偵測成功就自動恢復該分數。

### 2. 頸部估測／跨鏡頭品質對照

- 原本 v7 頸角保持不變（正式 RULA 評分沿用原值）。
- 另加入 v8 頸部相對軀幹的 2D 候選角、關鍵點來源與最低可見度品質指標。
- 左右兩鏡頭：只有雙方品質達標，且候選角差異不大時才建立**診斷用**加權融合候選值；差異太大或品質不足時顯示不融合原因。
- 會輸出 `Diagnostics/Neck_Comparison.csv` 與逐側報表欄位；角度更接近不代表人體真實量測更準確。
- 原始的雙側彙總頸角仍保留為舊版展示值，不參與重新推算每側 RULA。

### 3. Hand 診斷與原有功能

- 保留原有 Pose 2D / Pose 3D / Pose 混合 / Hand 2D 的同影格比較及雙照片。
- 新增 Hand 品質失敗原因統計，幫助追查左手偵測問題；**沒有提升模型匹配率的未經驗證宣稱**。
- 保留保守異常樣本警示、逐影格部位分數人工覆寫與重算、風險時間軸、左右側同步評分、原始圖片與關鍵點 JSONL 匯出。
- 左右分數相同時彙總表改顯示**「雙側同分」**，不再硬指定右側。

## ZIP 重要檔案

- `RULA_左右同步彙整報表.csv`、`RULA_右手側獨立報表.csv`、`RULA_左手側獨立報表.csv` 或 `RULA_Report.csv`
- `Diagnostics/Skipped_Samples.csv`
- `Diagnostics/Rejected_Samples_Detail.csv`
- `Diagnostics/Rejected_Images/`（圖片原圖＋實際重新推論的診斷圖）
- `Diagnostics/Rejected_Manual_Reviews.csv`（從最後一步下載的完整 ZIP）
- `Diagnostics/Neck_Comparison.csv`（雙側模式）
- `Diagnostics/Keypoint_Coordinates.jsonl`、`Raw_Frames/`（勾選匯出時）
- `Hand_Comparison/Comparison_Data.csv`
- 完整報表在有人工修正時，另外包含 `Diagnostics/V4_Manual_Overrides.csv` 等紀錄。

## 此版本做了哪些測試？

- `python -m py_compile` 語法檢查。
- 使用假的 MediaPipe 模型與合成 MP4 影片做**雙側、單側流程的離線模擬**：含成功樣本、剔除影格、照片／CSV／ZIP 對應及左右同分。
- 使用假的 Streamlit UI 跑過 12 個主步驟及分析後結果頁。
- 測試頸部候選融合：品質不足、左右差距太大時不融合；測試下載 ZIP 不覆蓋原始分數。
- **尚未在使用者實際的 Streamlit Cloud 與真實影片做端到端執行**。離線合成測試不代表角度真值、AI 辨識率或臨床／職業衛生評估準確性。
- Table A/B/C 沿用 v7.2，**尚未完成與官方 RULA 工作表逐格獨立核對**，專題正式驗證前仍應補做。

## 若部署出錯

- 嚴格使用 Python **3.11**，不要直接升級 MediaPipe 到 0.10.30+ 而未改寫 Pose Legacy API。
- OpenCV 載入失敗需確認 `packages.txt` 的 `libgl1`、`libglib2.0-0` 是否安裝成功；請提供 Cloud Logs 的最後一行 ImportError。
- 如果出現 `NameError`，先確認首頁版本標記與 GitHub `222.py` 是否一致。
