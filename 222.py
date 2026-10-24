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
                                "右手 Score D": data_r["頸軀腿 Score D"],
                                "右手 Score C": data_r["右手 Score C"],
                                "右手 Grand Score": r_g,
                                "右手 AL": data_r["右手 AL"],
                                "左手 Score A": data_l["左手 Score A"],
                                "左手 Score D": data_l["頸軀腿 Score D"],
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
                        skip_diagnostics.extend([{"樣本編號": i+1, "時間(秒)": sec, "評估側": side, "可能原因": "影片影格讀取失敗"} for side in ("右手側", "左手側")])

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
                zf.writestr("Diagnostics/Skipped_Samples.csv", pd.DataFrame(skip_diagnostics, columns=["樣本編號", "時間(秒)", "評估側", "可能原因"]).to_csv(index=False).encode("utf-8-sig"))
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
            skip_diagnostics = []
            
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
                            skip_diagnostics.append({"樣本編號": i+1, "時間(秒)": sec, "評估側": target_side, "可能原因": "正面關鍵點不足" if not front_ok else "側面關鍵點不足或角度無效"})

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
                        skip_diagnostics.append({"樣本編號": i+1, "時間(秒)": sec, "評估側": target_side, "可能原因": "影片影格讀取失敗"})

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
