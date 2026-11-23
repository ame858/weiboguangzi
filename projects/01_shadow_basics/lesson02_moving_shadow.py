"""第 2 课，生成带真值掩膜的运动阴影序列。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# 帧数组的形状是 (帧数, 高 y, 宽 x)，单位均为像素。
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 192
BACKGROUND_INTENSITY = 180.0

SHADOW_START_X = 8
SHADOW_START_Y = 48
SHADOW_WIDTH = 32
SHADOW_HEIGHT = 16
SHADOW_FACTOR = 0.25

FRAME_COUNT = 24
# 相邻两帧的时间间隔。教学假设值，实际系统帧率待实验室确认。
FRAME_INTERVAL_S = 0.2
# 每帧左上角移动的像素数。6 像素/帧 ÷ 0.2 秒 = 30 像素/秒。
SHADOW_STEP_X = 6
SHADOW_STEP_Y = 1


def validate_parameters():
    if IMAGE_HEIGHT <= 0 or IMAGE_WIDTH <= 0:
        raise ValueError("图像高、宽必须大于 0。")
    if SHADOW_WIDTH <= 0 or SHADOW_HEIGHT <= 0:
        raise ValueError("阴影宽、高必须大于 0。")
    if FRAME_COUNT <= 0:
        raise ValueError("帧数必须大于 0。")
    if FRAME_INTERVAL_S <= 0:
        raise ValueError("帧间隔必须大于 0。")
    if not 0 < BACKGROUND_INTENSITY <= 255:
        raise ValueError("背景强度必须大于 0 且不超过 255。")
    if not 0 <= SHADOW_FACTOR <= 1:
        raise ValueError("阴影强度保留比例必须位于 0 到 1 之间。")
    if SHADOW_STEP_X < 0 or SHADOW_STEP_Y < 0:
        raise ValueError("本课假设阴影向右下方运动，步长不能为负。")

    # 越界切片会让阴影悄悄变小，这里直接拦住。
    last_x = SHADOW_START_X + (FRAME_COUNT - 1) * SHADOW_STEP_X
    last_y = SHADOW_START_Y + (FRAME_COUNT - 1) * SHADOW_STEP_Y
    if not (0 <= SHADOW_START_X and last_x + SHADOW_WIDTH <= IMAGE_WIDTH):
        raise ValueError("阴影在横向上越出图像，请减小帧数或步长。")
    if not (0 <= SHADOW_START_Y and last_y + SHADOW_HEIGHT <= IMAGE_HEIGHT):
        raise ValueError("阴影在纵向上越出图像，请减小帧数或步长。")


def build_sequence():
    """返回帧数组、掩膜数组，以及每帧阴影左上角的轨迹。"""
    frames = np.empty((FRAME_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    masks = np.zeros((FRAME_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    trajectory = []
    for t in range(FRAME_COUNT):
        x = SHADOW_START_X + t * SHADOW_STEP_X
        y = SHADOW_START_Y + t * SHADOW_STEP_Y
        frame = np.full(
            (IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND_INTENSITY, dtype=np.float32
        )
        rows = slice(y, y + SHADOW_HEIGHT)
        columns = slice(x, x + SHADOW_WIDTH)
        frame[rows, columns] *= SHADOW_FACTOR
        masks[t, rows, columns] = 1
        frames[t] = frame
        trajectory.append((x, y))
    return frames, masks, trajectory


def centroids_from_masks(masks):
    """用掩膜的前景像素均值计算阴影中心，作为轨迹的独立检查。"""
    centroids = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        centroids.append((float(xs.mean()), float(ys.mean())))
    return centroids


def check_centroids(trajectory, centroids):
    for t, ((x, y), (cx, cy)) in enumerate(zip(trajectory, centroids)):
        expected_x = x + (SHADOW_WIDTH - 1) / 2
        expected_y = y + (SHADOW_HEIGHT - 1) / 2
        if not (np.isclose(cx, expected_x) and np.isclose(cy, expected_y)):
            raise RuntimeError(f"第 {t} 帧的阴影中心与设定轨迹不一致。")


def save_frame_preview(frames, output_dir):
    indices = np.linspace(0, FRAME_COUNT - 1, 6).round().astype(int)
    columns, rows = 3, 2
    gap, title_height = 10, 26
    canvas = Image.new(
        "RGB",
        (
            columns * IMAGE_WIDTH + (columns + 1) * gap,
            rows * (IMAGE_HEIGHT + title_height) + (rows + 1) * gap,
        ),
        "#20242b",
    )
    draw = ImageDraw.Draw(canvas)
    for index, t in enumerate(indices):
        column, row = index % columns, index // columns
        left = gap + column * (IMAGE_WIDTH + gap)
        top = gap + row * (IMAGE_HEIGHT + title_height + gap)
        draw.text(
            (left, top), f"t = {t}  ({t * FRAME_INTERVAL_S:.1f} s)", fill="white"
        )
        panel = Image.fromarray(np.rint(frames[t]).astype(np.uint8))
        canvas.paste(panel, (left, top + title_height))
    canvas.save(output_dir / "preview_frames.png")


def save_trajectory_preview(masks, centroids, output_dir):
    # 把各帧掩膜叠加，灰色区域表示阴影经过的位置。
    occupancy = masks.max(axis=0) * 60
    image = Image.fromarray(np.stack([occupancy] * 3, axis=-1))
    draw = ImageDraw.Draw(image)
    points = [(round(cx), round(cy)) for cx, cy in centroids]
    draw.line(points, fill=(255, 255, 255), width=1)
    for px, py in points:
        draw.rectangle([px - 1, py - 1, px + 1, py + 1], fill=(255, 255, 255))
    image = image.resize(
        (IMAGE_WIDTH * 2, IMAGE_HEIGHT * 2), Image.Resampling.NEAREST
    )
    image.save(output_dir / "preview_trajectory.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "day02",
        help="输出目录，重复运行会更新同名结果文件。",
    )
    args = parser.parse_args()

    validate_parameters()
    frames, masks, trajectory = build_sequence()
    centroids = centroids_from_masks(masks)
    check_centroids(trajectory, centroids)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "frames.npy", frames)
    np.save(output_dir / "masks.npy", masks)

    shadow_intensity = float(frames[0, SHADOW_START_Y, SHADOW_START_X])
    metadata = {
        "model": "moving_rectangular_intensity_attenuation",
        "intensity_unit": "arbitrary_linear_unit",
        "coordinate_unit": "pixel",
        "coordinate_convention": "frames[t, y, x]; x rightward; y downward; stop excluded",
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "background_intensity": BACKGROUND_INTENSITY,
        "frame_count": FRAME_COUNT,
        "frame_interval_s": FRAME_INTERVAL_S,
        "frame_interval_note": "教学假设值，实验室实际帧率待确认",
        "shadow_start": {"x": SHADOW_START_X, "y": SHADOW_START_Y},
        "shadow_width": SHADOW_WIDTH,
        "shadow_height": SHADOW_HEIGHT,
        "shadow_factor": SHADOW_FACTOR,
        "shadow_intensity": shadow_intensity,
        "shadow_pixels_per_frame": SHADOW_WIDTH * SHADOW_HEIGHT,
        "step_px_per_frame": {"x": SHADOW_STEP_X, "y": SHADOW_STEP_Y},
        "speed_px_per_s": {
            "x": SHADOW_STEP_X / FRAME_INTERVAL_S,
            "y": SHADOW_STEP_Y / FRAME_INTERVAL_S,
        },
        "trajectory": [
            {"frame": t, "x": x, "y": y} for t, (x, y) in enumerate(trajectory)
        ],
        "centroids": [
            {"frame": t, "cx": cx, "cy": cy} for t, (cx, cy) in enumerate(centroids)
        ],
        "array_shape": list(frames.shape),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    save_frame_preview(frames, output_dir)
    save_trajectory_preview(masks, centroids, output_dir)

    last_x, last_y = trajectory[-1]
    first_cx, first_cy = centroids[0]
    last_cx, last_cy = centroids[-1]
    print(f"帧数 {FRAME_COUNT}，帧间隔 {FRAME_INTERVAL_S:g} 秒")
    print(f"阴影尺寸（宽 × 高） {SHADOW_WIDTH} × {SHADOW_HEIGHT}")
    print(f"首帧左上角（x, y） ({SHADOW_START_X}, {SHADOW_START_Y})")
    print(f"末帧左上角（x, y） ({last_x}, {last_y})")
    print(f"首帧中心（x, y） ({first_cx:g}, {first_cy:g})")
    print(f"末帧中心（x, y） ({last_cx:g}, {last_cy:g})")
    print(f"阴影强度 {shadow_intensity:g}，单帧阴影像素 {SHADOW_WIDTH * SHADOW_HEIGHT}")
    print(
        f"速度（x, y） ({SHADOW_STEP_X / FRAME_INTERVAL_S:g}, "
        f"{SHADOW_STEP_Y / FRAME_INTERVAL_S:g}) 像素/秒"
    )
    print(f"输出目录 {output_dir}")


if __name__ == "__main__":
    main()
