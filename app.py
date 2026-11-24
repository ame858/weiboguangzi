"""Streamlit GUI for the Ka-band video SAR shadow project.

Start from the repository root:

    streamlit run app.py

The GUI is presentation-only: it shells out to ``python -m videosar`` and reads
existing result folders. It never modifies videosar/ or deliverables/.
"""
from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from videosar_ui import commands as cmd
from videosar_ui import imaging as viz
from videosar_ui import scanning as scan

# Newer Streamlit releases renamed use_container_width to width='stretch'.
# Probe the signature once so the GUI works on both 1.37+ and 1.6x+.
_WIDTH_KWARG = ({"width": "stretch"}
                if "width" in inspect.signature(st.image).parameters
                else {"use_container_width": True})


def display_settings():
    """Shared display controls. Explicitly labelled as visual-only."""
    with st.sidebar.expander("显示增强（仅用于视觉增强，不改变底层检测输入）", expanded=True):
        mode = st.selectbox("灰度映射模式", viz.MODES,
                            format_func=lambda key: viz.MODE_LABELS[key], key="disp_mode")
        cmap = st.selectbox("伪彩色", viz.CMAPS, key="disp_cmap")
        denoise = st.checkbox("轻度显示去斑平滑（仅预览）", value=False, key="disp_denoise")
        diff_ref = st.selectbox("时序差分参考", ("median", "previous"),
                                format_func=lambda key: "短期历史中值" if key == "median" else "前一帧",
                                key="disp_diff")
        show_truth = st.checkbox("叠加绿色真值轮廓", value=True, key="disp_truth")
        show_pred = st.checkbox("叠加红色预测掩膜", value=True, key="disp_pred")
    return {"mode": mode, "cmap": cmap, "denoise": denoise, "diff_ref": diff_ref,
            "show_truth": show_truth, "show_pred": show_pred}


def build_views(sequence, predictions, track_rows, index, settings):
    frames = sequence["frames"]
    truth = sequence["masks"]
    prediction = None if predictions is None else predictions[index]
    base = viz.enhance(frames[index], settings["mode"], settings["cmap"], settings["denoise"])

    view1 = base.copy()
    if settings["show_truth"] and truth is not None:
        view1 = viz.overlay_mask(view1, truth[index], viz.TRUTH_COLOR)
    if settings["show_pred"] and prediction is not None:
        view1 = viz.overlay_mask(view1, prediction, viz.PREDICTION_COLOR)

    view2 = viz.temporal_difference(frames, index, settings["diff_ref"])

    view3 = base.copy()
    if prediction is not None:
        view3 = viz.overlay_mask(view3, prediction, viz.PREDICTION_COLOR)
    view3 = viz.draw_tracks(view3, track_rows, index)

    center = viz.centroid(prediction)
    if center is None and truth is not None:
        center = viz.centroid(truth[index])
    view4, _roi = viz.roi_zoom(view1, center, factor=4)
    return [view1, view2, view3, view4], center


def _run(commands, key):
    lines, box = [], st.empty()

    def emit(text):
        lines.append(str(text))
        box.code("\n".join(lines[-60:]))

    ok = cmd.run_commands(commands, scan.REPO_ROOT, emit)
    if ok:
        st.success("运行完成，结果已写入 runs/gui/ 下的新目录。")
    else:
        st.error("命令执行失败，请查看上方实时日志；不会覆盖任何已有实验存档。")
    return ok


def _pick_dir(candidates, label, key, preferred=None):
    options = list(dict.fromkeys([str(path) for path in ([preferred] if preferred else []) + list(candidates)]))
    labels = ["（未选择）"] + options
    index = 1 if options else 0
    chosen = st.selectbox(label, range(len(labels)), index=index,
                          format_func=lambda i: labels[i], key=f"{key}_select")
    manual = st.text_input(f"{label} · 或手动填写绝对路径", key=f"{key}_manual",
                           placeholder=r"例如 E:\workspace\weiboguangzi\runs\live-detect")
    if manual.strip():
        return Path(manual.strip())
    return Path(options[chosen - 1]) if chosen else None


def render_browser(sequence_dir, results, settings, key):
    """Four-panel frame browser plus dynamic KPI cards and track charts."""
    sequence = scan.load_sequence(sequence_dir)
    if sequence is None:
        st.error(f"无法加载序列：{sequence_dir}（需要 frames.npy 与 metadata.json）")
        return
    frames = sequence["frames"]
    count = len(frames)
    predictions = scan.load_predictions(results.get("predictions"), frames.shape)
    tracks = scan.load_tracks(results.get("tracks"))
    metrics = scan.load_metrics(results.get("metrics"))

    missing = [name for name, folder in (("predictions.npy", results.get("predictions")),
                                         ("tracks.csv", results.get("tracks")),
                                         ("metrics.json", results.get("metrics"))) if not folder]
    if missing or metrics is None:
        st.info("缺少 " + "、".join(missing or ["metrics.json"]) +
                "：相关指标显示为 —。可在模式 B 现场运行完整链路，或用上方输入框手动指定目录。")

    st.markdown("#### 关键指标（全部从 metrics.json / tracks.json 动态读取）")
    cards = scan.metric_cards(metrics, count)
    columns = st.columns(5)
    for position, (label, value) in enumerate(cards):
        columns[position % 5].metric(label, scan.format_metric(value))
    summary = scan.tracking_summary(metrics, tracks)
    columns = st.columns(3)
    for position, (label, value) in enumerate(summary.items()):
        columns[position % 3].metric(label, scan.format_metric(value))
    if metrics and metrics.get("match_iou") is not None:
        st.caption(f"目标级匹配阈值 IoU ≥ {scan.format_metric(metrics['match_iou'])}；"
                   f"指标来源：{metrics['path']}")

    st.markdown("#### 四宫格同步帧浏览器")
    st.caption("视图 1 增强 SAR 原图（绿色真值 / 红色预测轮廓） · 视图 2 时序差分（静止雪花与静态暗区被抵消） · "
               "视图 3 检测与轨迹（青色 measured / 橙色 imputed） · 视图 4 跟随阴影质心的 4 倍 ROI 放大")
    index = st.slider("统一帧号", 0, max(0, count - 1), 0, key=f"{key}_frame") if count > 1 else 0
    views, center = build_views(sequence, predictions, tracks["rows"], index, settings)
    st.image(viz.compose_quad(views),
             caption=f"帧 {index} / {count - 1}" + (f" · 质心 ({center[0]:.1f}, {center[1]:.1f})" if center else " · 本帧无阴影质心"),
             **_WIDTH_KWARG)

    control_a, control_b, control_c = st.columns([2, 2, 3])
    with control_a:
        play = st.button("播放四宫格动画", key=f"{key}_play")
    with control_b:
        export_gif = st.button("导出 GIF 动图", key=f"{key}_gif")
    with control_c:
        export_mp4 = st.button("导出 MP4（需 imageio-ffmpeg）", key=f"{key}_mp4")

    if play:
        placeholder = st.empty()
        for frame_index in range(count):
            panel_views, _ = build_views(sequence, predictions, tracks["rows"], frame_index, settings)
            placeholder.image(viz.compose_quad(panel_views), caption=f"播放中：帧 {frame_index} / {count - 1}",
                              **_WIDTH_KWARG)
            time.sleep(0.15)
        st.success("播放结束。")

    if export_gif or export_mp4:
        output = cmd.gui_run_dir("export")
        with st.spinner("正在生成四宫格动画 ..."):
            panels = [viz.compose_quad(build_views(sequence, predictions, tracks["rows"], i, settings)[0])
                      for i in range(count)]
            if export_gif:
                path = viz.save_gif(panels, output / "quad.gif", duration_ms=125)
                st.success(f"已保存：{path}")
                st.download_button("下载四宫格 GIF", path.read_bytes(),
                                   file_name="sar_quad.gif", mime="image/gif", key=f"{key}_dl_gif")
            else:
                try:
                    path = viz.save_mp4(panels, output / "quad.mp4", fps=8)
                    st.success(f"已保存：{path}")
                    st.download_button("下载四宫格 MP4", path.read_bytes(),
                                       file_name="sar_quad.mp4", mime="video/mp4", key=f"{key}_dl_mp4")
                except (ImportError, OSError, RuntimeError) as error:
                    st.error(f"MP4 导出失败：{error}。请安装 imageio-ffmpeg 或改用 GIF。")

    if tracks["rows"]:
        st.markdown("#### 逐帧轨迹坐标与表观速度")
        frame_indices, speeds = scan.per_frame_speed(tracks["rows"])
        coordinates = pd.DataFrame({
            "帧号": frame_indices,
            "x_px": [row["x_px"] for row in tracks["rows"]],
            "y_px": [row["y_px"] for row in tracks["rows"]],
        }).set_index("帧号")
        st.line_chart(coordinates, height=180)
        st.line_chart(pd.DataFrame({"帧号": frame_indices, "表观速度 (px/s)": speeds}).set_index("帧号"),
                      height=160)
        st.dataframe(pd.DataFrame(tracks["rows"]), **_WIDTH_KWARG)
        st.caption(f"来源：{tracks.get('path')}")


@st.cache_data(ttl=15, show_spinner=False)
def _sequences_cached(root_text):
    return [str(path) for path in scan.find_sequences(Path(root_text))]


@st.cache_data(ttl=15, show_spinner=False)
def _results_cached(root_text, sequence_text):
    raw = scan.find_results(Path(root_text), Path(sequence_text))
    return {kind: ([str(path) for path in value] if isinstance(value, list) else value)
            for kind, value in raw.items()}


def sequences_for(runs_path):
    """Cached scan, so a large runs/ tree is not walked on every interaction."""
    return [Path(text) for text in _sequences_cached(str(runs_path))]


def _sequence_label(path, runs_root):
    text = scan._relative_text(path, runs_root)
    if text.endswith(scan.PREFERRED_SEQUENCE):
        return f"[结题演示·优先] {text}"
    if "gui/" in text or text.startswith("gui"):
        return f"[GUI 现场结果] {text}"
    return text


def _results_for(runs_path, sequence_dir):
    with st.spinner("正在扫描该序列的结果目录 ..."):
        raw = _results_cached(str(runs_path), str(sequence_dir))
    results = {kind: ([Path(text) for text in value] if isinstance(value, list) else value)
               for kind, value in raw.items()}
    live = scan.live_dirs(runs_path)
    preferred = {"predictions": live.get("live-detect"),
                 "tracks": live.get("live-track"), "metrics": live.get("live-eval")}
    for kind, folder in preferred.items():
        if folder and folder.is_dir():
            results[kind] = [folder] + [path for path in results[kind] if path != folder]
    return results


def tab1_mode_a(runs_path, settings):
    st.markdown("##### 模式 A · 秒播已有结果（默认）")
    sequences = sequences_for(runs_path)
    manual = st.text_input("手动指定序列目录（绝对路径，留空则用下方扫描结果）", key="a_manual_seq")
    if manual.strip():
        sequence_dir = Path(manual.strip())
    elif sequences:
        choice = st.selectbox("扫描到的 frames.npy 数据目录", range(len(sequences)),
                              format_func=lambda i: _sequence_label(sequences[i], runs_path), key="a_seq")
        sequence_dir = sequences[choice]
    else:
        st.warning(f"未在 {runs_path} 找到包含 frames.npy 的目录（该目录可能被 .gitignore 排除或尚未生成）。")
        st.markdown("处理方式：① 在上方手动填入已有结果路径；② 一键生成快速演示；③ 切到模式 B 现场运行。")
        if st.button("一键生成快速图像域演示", key="a_bootstrap"):
            output = cmd.gui_run_dir("bootstrap")
            demo_dir = output / "demo"
            if _run([cmd.demo_command(cmd.DEMO_CONFIG, 16, demo_dir)], "a_boot"):
                render_browser(demo_dir / "sequence",
                               {"predictions": demo_dir / "detect", "tracks": demo_dir / "track",
                                "metrics": demo_dir / "evaluate"}, settings, "a_boot_browser")
        return

    if not sequence_dir.is_dir() or not (sequence_dir / "frames.npy").is_file():
        st.error(f"目录无效或缺少 frames.npy：{sequence_dir}")
        return
    results = _results_for(runs_path, sequence_dir)
    if not any(results["_linked"].values()):
        st.caption("未找到与所选序列通过 source.json 关联的结果目录，以下为全部候选，请手动核对。"
                   "如已按 README 生成 runs/live-detect / live-track / live-eval，它们已优先置顶。")
    with st.expander("结果目录（检测 / 跟踪 / 评估）", expanded=True):
        left, middle, right = st.columns(3)
        with left:
            detect_dir = _pick_dir(results["predictions"], "检测结果目录", "a_detect")
        with middle:
            track_dir = _pick_dir(results["tracks"], "跟踪结果目录", "a_track")
        with right:
            evaluate_dir = _pick_dir(results["metrics"], "评估结果目录", "a_eval")
    render_browser(sequence_dir, {"predictions": detect_dir, "tracks": track_dir,
                                  "metrics": evaluate_dir}, settings, "tab1_a")


def tab1_mode_b(settings):
    st.markdown("##### 模式 B · 现场实时运行")
    st.caption("所有现场产物写入 runs/gui/<时间戳_任务名>/，绝不覆盖已有 runs/ 存档。")
    choice = st.radio("运行方式", ("快速图像域演示（推荐现场用）", "单通道 SAR 回波成像与检测"),
                      key="b_choice")
    if choice.startswith("快速"):
        config = st.text_input("配置 JSON", value=str(cmd.DEMO_CONFIG), key="b_config")
        frames = st.slider("帧数", 8, 24, 16, step=1, key="b_frames")
        st.code(f"{Path(sys.executable).name} -m videosar demo --config {config} --frames {frames}")
        if st.button("运行快速演示", key="b_run_demo"):
            output = cmd.gui_run_dir("demo")
            demo_dir = output / "demo"
            if _run([cmd.demo_command(config, frames, demo_dir)], "b_demo_log"):
                st.markdown("##### 运行结果 · 四宫格浏览器")
                render_browser(demo_dir / "sequence",
                               {"predictions": demo_dir / "detect", "tracks": demo_dir / "track",
                                "metrics": demo_dir / "evaluate"}, settings, "b_demo_browser")
    else:
        bandwidth = st.selectbox("带宽 (bandwidth-hz)", (300000000, 450000000, 600000000),
                                 format_func=lambda value: f"{value / 1e6:.0f} MHz", key="b_bw")
        method = st.selectbox("检测方法", ("threshold", "difference", "learned"),
                              format_func=lambda value: {"threshold": "传统：局部暗度阈值",
                                                         "difference": "传统：时序背景差分",
                                                         "learned": "学习模型：冻结 U-Net（需 torch）"}[value],
                              key="b_method")
        model = cmd.UNET_MODEL
        if method == "learned":
            model = Path(st.text_input("模型部署 JSON", value=str(cmd.UNET_MODEL), key="b_model"))
        st.code("image -> detect -> track -> evaluate（--match-iou 0.50）")
        if st.button("运行 SAR 回波成像与检测全链路", key="b_run_sar"):
            output = cmd.gui_run_dir("sar")
            plan = cmd.sar_chain_commands(bandwidth, method, model, output)
            if _run(plan["commands"], "b_sar_log"):
                st.markdown("##### 运行结果 · 四宫格浏览器")
                render_browser(plan["sequence"], plan, settings, "b_sar_browser")


def render_tab1(runs_path, settings):
    st.subheader("Tab 1 · 仿真与结题闭环演示")
    mode = st.radio("演示模式", ("模式 A：秒播已有结果（默认）", "模式 B：现场实时运行"),
                    horizontal=True, key="tab1_mode")
    if mode.startswith("模式 A"):
        tab1_mode_a(runs_path, settings)
    else:
        tab1_mode_b(settings)


def _numeric_frame(rows):
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() >= max(1, int(0.8 * len(frame))):
            frame[column] = converted
    return frame


def render_tab2(runs_path, settings):
    st.subheader("Tab 2 · 算法与物理条件同屏对照")
    sequences = sequences_for(runs_path)
    if not sequences:
        st.warning(f"未在 {runs_path} 找到 frames.npy 序列，请先在 Tab 1 生成结果或手动填写路径。")
        manual = st.text_input("序列目录绝对路径", key="t2_manual_seq")
        if not manual.strip():
            return
        sequences = [Path(manual.strip())]
    choice = st.selectbox("同源 frames.npy 序列", range(len(sequences)),
                          format_func=lambda i: _sequence_label(sequences[i], runs_path), key="t2_seq")
    sequence_dir = sequences[choice]
    sequence = scan.load_sequence(sequence_dir)
    if sequence is None:
        st.error(f"无法加载序列：{sequence_dir}")
        return
    frames = sequence["frames"]

    results = _results_for(runs_path, sequence_dir)
    candidates = [str(path) for path in results["predictions"]]
    candidates += [path for path in st.session_state.get("t2_extra", []) if path not in candidates]
    if st.button("对当前序列一键跑：传统基线 vs 学习模型", key="t2_compare_run"):
        output = cmd.gui_run_dir("compare")
        plan = cmd.compare_commands(sequence_dir, output)
        if _run(plan["commands"], "t2_compare_log"):
            st.session_state["t2_extra"] = [str(path) for path in plan["predictions"]]
            candidates = [str(path) for path in plan["predictions"]] + candidates
    if not candidates:
        st.info("未找到任何 predictions.npy 目录。可点击上方按钮现跑两组检测，或在 Tab 1 模式 B 生成。")
        return

    left_col, right_col = st.columns(2)
    with left_col:
        left = st.selectbox("左侧检测目录", candidates, key="t2_left")
    with right_col:
        right = st.selectbox("右侧检测目录", candidates, index=min(1, len(candidates) - 1), key="t2_right")
    index = st.slider("同屏帧号", 0, max(0, len(frames) - 1), 0, key="t2_frame") if len(frames) > 1 else 0
    truth = sequence["masks"]
    base = viz.enhance(frames[index], settings["mode"], settings["cmap"], settings["denoise"])

    for column, path, title in zip((left_col, right_col), (left, right), ("左侧", "右侧")):
        prediction = scan.load_predictions(Path(path), frames.shape)
        view = base.copy()
        if prediction is not None:
            view = viz.overlay_mask(view, prediction[index], viz.PREDICTION_COLOR)
        if truth is not None:
            view = viz.overlay_mask(view, truth[index], viz.TRUTH_COLOR)
        column.image(view, caption=f"{title}：{path}", **_WIDTH_KWARG)
        components = viz.count_components(prediction[index]) if prediction is not None else None
        overlap = (int(np.count_nonzero(prediction[index] & truth[index]))
                   if prediction is not None and truth is not None else None)
        metric_a, metric_b = column.columns(2)
        metric_a.metric("预测连通域数（预览计数）", scan.format_metric(components))
        metric_b.metric("与真值重叠像素", scan.format_metric(overlap))
        column.caption("虚警数以对应 evaluate/metrics.json 为准；红色=预测，绿色=真值轮廓。")
    if left == right:
        st.caption("提示：左右选择了同一目录，可在上方改成两个不同的检测结果。")

    st.markdown("#### 归档指标表（自动从 deliverables/ 读取）")
    tables = scan.archive_tables()
    if not tables:
        st.info("未找到归档 CSV。")
        return
    labels = [label for label, _ in tables]
    selected = st.selectbox("选择归档指标表", range(len(tables)),
                            format_func=lambda i: labels[i], key="t2_table")
    label, path = tables[selected]
    fields, rows = scan.read_csv_rows(path)
    if not rows:
        st.warning(f"{path} 无可读取的数据行。")
        return
    frame = _numeric_frame(rows)
    st.caption(f"{label} · 来源：{path}")
    st.dataframe(frame, **_WIDTH_KWARG)
    try:
        if "condition" in frame.columns and "recall" in frame.columns and "method" in frame.columns:
            pivot = frame.pivot_table(index="condition", columns="method", values="recall", aggfunc="mean")
            st.markdown("分条件召回率对比")
            st.bar_chart(pivot)
        elif "method" in frame.columns:
            metrics = [name for name in ("precision", "recall", "target_f1", "region_iou",
                                         "false_positive_per_frame") if name in frame.columns]
            if metrics:
                st.bar_chart(frame.set_index("method")[metrics].apply(pd.to_numeric, errors="coerce"))
    except (KeyError, ValueError, TypeError) as error:
        st.caption(f"柱状图跳过：{error}")


def render_tab3(runs_path, settings):
    st.subheader("Tab 3 · 实测雷达视频推理（detect-video）")
    info = scan.snl_model_info()
    development = info.get("development") or {}
    st.info("客观说明：随附七帧模型为**严格因果、无未来帧泄露**的探索性研究权重，"
            "仅用于研究演示。训练/校准/开发复核取自 SNL Eubank Gate 开发段，"
            "两套正式验收段仍处于封存状态；下方开发复核成绩不代表正式验收结果。")
    if info:
        st.markdown("#### 模型信息（自动读取 stacked-center.json）")
        fields = {key: info.get(key) for key in
                  ("architecture", "method", "input_channels", "epoch", "threshold", "nms",
                   "input_profile", "validation_scope")}
        st.dataframe(pd.DataFrame([{"字段": key, "值": str(value)} for key, value in fields.items()]),
                     **_WIDTH_KWARG, hide_index=True)
        if development:
            detail = pd.DataFrame([{"指标": label, "数值": scan.format_metric(development.get(key))}
                                   for label, key in (("开发复核帧数", "frames"), ("命中 TP", "tp"),
                                                      ("误报 FP", "fp"), ("漏检 FN", "fn"),
                                                      ("查准率", "precision"), ("召回率", "recall"),
                                                      ("F1", "f1"), ("每帧虚警", "fp_per_frame"))])
            st.dataframe(detail, **_WIDTH_KWARG, hide_index=True)

    previews = scan.preview_videos(runs_path)
    st.markdown("#### 预渲染预览")
    if previews:
        selected = st.selectbox("选择已归档演示视频", previews, key="t3_preview")
        st.video(selected)
    else:
        st.caption("未找到 runs/snl-causal-study/prospective/development-preview.mp4 或 "
                   "runs/snl-video-prefix-v2/overlay.mp4，可先运行 detect-video 生成 overlay.mp4。")

    environment = cmd.environment_report()
    st.markdown("#### 环境自检")
    environment_col, hint_col = st.columns(2)
    with environment_col:
        st.write({"FFmpeg": environment["ffmpeg"] or "未找到", "PyTorch": environment["torch"],
                  "torchvision": environment["torchvision"], "imageio-ffmpeg": environment["imageio_ffmpeg"],
                  "Python": environment["python"]})
    with hint_col:
        if not environment["ffmpeg"]:
            st.error("缺少 FFmpeg。安装命令：" + cmd.FFMPEG_INSTALL)
        if not environment["torch"]:
            st.error("缺少 PyTorch。安装命令：" + cmd.TORCH_INSTALL)
        if environment["ffmpeg"] and environment["torch"]:
            st.success("FFmpeg 与 PyTorch 均可用。")

    st.markdown("#### 自定义视频推理")
    uploaded = st.file_uploader("上传视频（.mp4）", type=("mp4", "mov", "avi", "mkv"), key="t3_upload")
    local_path = st.text_input("或填写本地视频绝对路径", key="t3_path",
                               placeholder=r"例如 D:\data\radar.mp4")
    video_path = None
    workspace = None
    if uploaded is not None:
        workspace = cmd.gui_run_dir("video-input")
        workspace.mkdir(parents=True, exist_ok=True)
        video_path = workspace / uploaded.name
        video_path.write_bytes(uploaded.getvalue())
        st.caption(f"已暂存上传视频：{video_path}")
    elif local_path.strip():
        video_path = Path(local_path.strip())
    if video_path is not None and not video_path.is_file():
        st.error(f"视频文件不存在：{video_path}")
        video_path = None

    crop = (0, 0, 0, 0)
    if video_path is not None and environment["ffmpeg"]:
        if workspace is None:
            workspace = cmd.gui_run_dir("video-preview")
        first = cmd.extract_first_frame(video_path, environment["ffmpeg"], workspace / "first_frame.png")
        if first is None:
            st.warning("无法从该视频解码首帧，请确认 FFmpeg 可读取该文件。")
        else:
            with Image.open(first) as image:
                rgb = np.asarray(image.convert("RGB"))
            height, width = rgb.shape[:2]
            st.caption(f"首帧尺寸：{width} x {height}。下方红框即 --crop X Y WIDTH HEIGHT 的实时预览。")
            left, middle = st.columns(2)
            with left:
                x = st.number_input("X", 0, max(0, width - 2), 0, key="t3_x")
                y = st.number_input("Y", 0, max(0, height - 2), 0, key="t3_y")
            with middle:
                crop_w = st.number_input("WIDTH", 2, max(2, width), max(2, width), key="t3_w")
                crop_h = st.number_input("HEIGHT", 2, max(2, height), max(2, height), key="t3_h")
            if x + crop_w > width or y + crop_h > height:
                st.warning("裁剪框超出首帧范围，请减小 WIDTH / HEIGHT 或移动 X / Y；"
                           "不传 --crop 时默认使用整幅画面。")
            st.image(viz.draw_crop_box(rgb, (x, y, crop_w, crop_h)), **_WIDTH_KWARG)
            use_crop = st.checkbox("将该裁剪框传给 --crop", value=False, key="t3_use_crop")
            crop = (x, y, crop_w, crop_h) if use_crop else None
    else:
        crop = None

    left, right = st.columns(2)
    with left:
        start_frame = st.number_input("--start-frame（零基源帧）", 0, 100000, 0, key="t3_start")
    with right:
        max_frames = st.number_input("--max-frames", 1, 2000, 24, key="t3_max")
    st.caption("CPU 推理 378 帧约需 3–4 分钟；现场演示建议先跑 20–40 帧（默认 24 帧）。")
    model_path = st.text_input("模型部署 JSON（留空使用项目默认七帧权重）", value="", key="t3_model")

    if st.button("开始推理 detect-video", key="t3_run"):
        if video_path is None:
            st.error("请先上传视频或填写本地视频路径。")
        elif not environment["ffmpeg"] or not environment["torch"]:
            st.error("依赖未就绪，请按上方命令安装后重启 GUI。")
        else:
            output = cmd.gui_run_dir("video")
            command = cmd.detect_video_command(
                video_path, output, crop=crop, start_frame=start_frame, max_frames=max_frames,
                model=Path(model_path.strip()) if model_path.strip() else None,
                ffmpeg=environment["ffmpeg"])
            if _run([command], "t3_log"):
                overlay = output / "overlay.mp4"
                predictions = output / "predictions.csv"
                if overlay.is_file():
                    st.video(str(overlay))
                    st.download_button("下载 overlay.mp4", overlay.read_bytes(),
                                       file_name="overlay.mp4", mime="video/mp4", key="t3_dl_video")
                if predictions.is_file():
                    fields, rows = scan.read_csv_rows(predictions)
                    st.markdown("#### predictions.csv 预览")
                    st.dataframe(pd.DataFrame(rows), **_WIDTH_KWARG)
                    st.download_button("下载 predictions.csv", predictions.read_bytes(),
                                       file_name="predictions.csv", mime="text/csv", key="t3_dl_csv")
                video_info = scan.read_json_safe(output / "video.json")
                if video_info:
                    st.json(video_info)


def main():
    st.set_page_config(page_title="Ka 视频 SAR 阴影检测演示", layout="wide")
    st.title("Ka 频段视频 SAR 动目标阴影仿真与检测 · 结题演示台")
    st.caption("四宫格雷达可视化 / 显示增强仅用于视觉观察，不改变底层检测输入；"
               "所有指标均从结果目录的 JSON 与 CSV 动态读取。")
    with st.sidebar:
        st.header("全局设置")
        runs_text = st.text_input("结果根目录 (runs)", value=str(scan.DEFAULT_RUNS), key="runs_root")
        runs_path = Path(runs_text.strip()).expanduser()
        st.caption(f"仓库根目录：{scan.REPO_ROOT}")
        if not runs_path.is_dir():
            st.warning(f"当前结果根目录不存在：{runs_path}。可手动填写已有 runs 路径，或先运行一次演示生成。")
        st.divider()
        settings = display_settings()
        st.divider()
        st.caption("运行入口：streamlit run app.py；现场产物统一写入 runs/gui/。")

    tab1, tab2, tab3 = st.tabs(["Tab 1  仿真与结题闭环演示", "Tab 2  算法与物理条件同屏对照",
                                "Tab 3  实测雷达视频推理"])
    with tab1:
        render_tab1(runs_path, settings)
    with tab2:
        render_tab2(runs_path, settings)
    with tab3:
        render_tab3(runs_path, settings)


if __name__ == "__main__":
    main()
