"""读回一次 SAR 成像结果，检查分辨率关系与"几何遮挡"和"成像可见"的差别。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(run_dir):
    frames = np.load(run_dir / "frames.npy")
    visible = np.load(run_dir / "masks.npy").astype(bool)
    geometric = np.load(run_dir / "geometric_masks.npy").astype(bool)
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    return frames, visible, geometric, meta


def resolution_report(meta):
    """把三个容易混淆的量摆在一起：采样间隔、标称分辨率、采样率。"""
    bin_m = meta["slant_range_bin_m"]
    res_m = meta["nominal_slant_resolution_m"]
    fs_hz = meta["sample_rate_hz"]
    bw_hz = meta["effective_config"]["bandwidth_hz"]
    print(f"带宽 B            {bw_hz / 1e6:.0f} MHz")
    print(f"采样率 2B         {fs_hz / 1e9:.2f} GSa/s")
    print(f"距离 bin 间距     {bin_m:.4f} m   (c/4B)")
    print(f"标称距离分辨率    {res_m:.4f} m   (c/2B)")
    print(f"分辨率 / bin      {res_m / bin_m:.2f}   每个分辨率单元被采 {res_m / bin_m:.0f} 次")
    if not np.isclose(res_m / bin_m, 2.0, atol=1e-6):
        raise RuntimeError("分辨率与 bin 间距的 2 倍关系不成立，检查采样约定。")
    return bin_m, res_m


def contrast_table(frames, visible, geometric):
    """逐帧比较几何遮挡面积、成像可见面积和阴影内外的功率差。"""
    rows = []
    for t, (frame, vis, geo) in enumerate(zip(frames, visible, geometric)):
        if not vis.any():
            rows.append((t, int(geo.sum()), 0, 0.0, 0.0))
            continue
        inside = float(frame[vis].mean())
        outside = float(frame[~vis].mean())
        contrast_db = 10 * np.log10(inside / outside)
        rows.append((t, int(geo.sum()), int(vis.sum()), contrast_db, vis.sum() / max(geo.sum(), 1)))
    print("\n帧  几何px  可见px  可见率   阴影对比度")
    for t, geo_px, vis_px, db, ratio in rows:
        print(f"{t:2d}  {geo_px:7d}  {vis_px:7d}  {ratio:6.0%}  {db:+9.2f} dB")
    visible_frames = sum(1 for row in rows if row[2] > 0)
    print(f"\n几何遮挡帧 {len(rows)}，其中有成像可见掩膜的 {visible_frames} 帧，"
          f"成像可见率 {visible_frames / len(rows):.0%}")
    return rows


def erode(mask):
    """3x3 腐蚀，用位移与运算实现，不依赖 scipy。"""
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def render(frame, mask, scale=4):
    """功率映射到 dB 后固定区间显示，保证不同帧之间亮度可比。"""
    db = 10 * np.log10(np.maximum(frame, 1e-8))
    clipped = np.clip((db + 25) / 30, 0, 1)
    gray = (clipped * 255).astype(np.uint8)
    rgb = np.stack([gray] * 3, axis=-1)
    outline = mask & ~erode(mask)
    rgb[outline] = [255, 214, 102]
    panel = Image.fromarray(rgb).resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.Resampling.NEAREST
    )
    return panel


def font(size):
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_preview(frames, visible, geometric, output_path, picks):
    scale = 4
    height, width = frames.shape[1:]
    panel_w, panel_h = width * scale, height * scale
    title_h, gap = 26, 12
    columns, rows = 2, 2
    canvas = Image.new(
        "RGB",
        (columns * panel_w + (columns + 1) * gap,
         rows * (panel_h + title_h) + (rows + 1) * gap),
        "#20242b",
    )
    draw = ImageDraw.Draw(canvas)
    small = font(14)
    for index, t in enumerate(picks):
        column, row = index % columns, index // columns
        left = gap + column * (panel_w + gap)
        top = gap + row * (panel_h + title_h + gap)
        geo_px = int(geometric[t].sum())
        vis_px = int(visible[t].sum())
        label = f"t={t}  geometric {geo_px} px / visible {vis_px} px"
        draw.text((left, top + 4), label, fill="#e8e8e8", font=small)
        panel = render(frames[t], visible[t], scale)
        canvas.paste(panel, (left, top + title_h))
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="videosar image 生成的运行目录")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()

    run_dir = args.run.resolve()
    frames, visible, geometric, meta = load(run_dir)
    print(f"运行目录 {run_dir}")
    print(f"帧数组形状 {frames.shape}   dtype {frames.dtype}")
    print(f"功率范围 {frames.min():.2e} — {frames.max():.2e}，均值 {frames.mean():.3f}\n")

    resolution_report(meta)
    rows = contrast_table(frames, visible, geometric)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # 取对比度最强和最弱的两帧，方便直接比较"挡得住"和"看得见"。
    ranked = sorted(rows, key=lambda row: row[3])
    picks = [ranked[0][0], ranked[len(ranked) // 2][0], ranked[-1][0], rows[len(rows) // 2][0]]
    picks = list(dict.fromkeys(picks))[:4]
    preview = output_dir / f"{run_dir.name}_inspect.png"
    build_preview(frames, visible, geometric, preview, picks)
    print(f"\n对比图 {preview}（黄框为成像可见掩膜边界）")


if __name__ == "__main__":
    main()
