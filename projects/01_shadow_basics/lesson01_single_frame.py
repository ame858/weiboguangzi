"""第 1 课，生成单帧图像域暗区和真值掩膜。"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# 数组先写行 y，再写列 x。这里的尺寸和位置单位均为像素。
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 192
BACKGROUND_INTENSITY = 180.0

SHADOW_X = 72
SHADOW_Y = 48
SHADOW_WIDTH = 48
SHADOW_HEIGHT = 24
# 暗区保留的强度比例，数值越大，暗区越亮。
SHADOW_FACTOR = 0.25


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "day01",
        help="输出目录，重复运行会更新同名结果文件。",
    )
    args = parser.parse_args()

    # 先检查参数，避免越界切片悄悄缩小阴影区域。
    shadow_right = SHADOW_X + SHADOW_WIDTH
    shadow_bottom = SHADOW_Y + SHADOW_HEIGHT
    if not (IMAGE_HEIGHT > 0 and IMAGE_WIDTH > 0):
        raise ValueError("图像高、宽必须大于 0。")
    if not (SHADOW_WIDTH > 0 and SHADOW_HEIGHT > 0):
        raise ValueError("暗区宽、高必须大于 0。")
    if not (0 <= SHADOW_X < shadow_right <= IMAGE_WIDTH):
        raise ValueError("暗区的横坐标范围必须位于图像内。")
    if not (0 <= SHADOW_Y < shadow_bottom <= IMAGE_HEIGHT):
        raise ValueError("暗区的纵坐标范围必须位于图像内。")
    if not (0 < BACKGROUND_INTENSITY <= 255):
        raise ValueError("本课背景强度必须大于 0 且不超过 255。")
    if not (0 <= SHADOW_FACTOR <= 1):
        raise ValueError("暗区强度保留比例必须位于 0 到 1 之间。")

    # 1. 创建背景，复制后只改变暗区内的强度。
    background = np.full(
        (IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND_INTENSITY, dtype=np.float32
    )
    image = background.copy()
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)

    # 2. 终点不包含在切片内，选中区域共有宽 × 高个像素。
    shadow_slice = np.s_[SHADOW_Y:shadow_bottom, SHADOW_X:shadow_right]
    image[shadow_slice] *= SHADOW_FACTOR
    mask[shadow_slice] = 1

    # 3. 保存原始数组。真值由生成过程给出，后续用于评价检测结果。
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "image.npy", image)
    np.save(output_dir / "mask.npy", mask)

    shadow_pixels = int(mask.sum())
    metadata = {
        "model": "uniform_background_with_rectangular_intensity_attenuation",
        "intensity_unit": "arbitrary_linear_unit",
        "coordinate_unit": "pixel",
        "coordinate_convention": "array[y, x]; x rightward; y downward; stop excluded",
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "background_intensity": BACKGROUND_INTENSITY,
        "shadow_x": SHADOW_X,
        "shadow_y": SHADOW_Y,
        "shadow_width": SHADOW_WIDTH,
        "shadow_height": SHADOW_HEIGHT,
        "shadow_factor": SHADOW_FACTOR,
        "shadow_intensity": float(image[SHADOW_Y, SHADOW_X]),
        "shadow_pixels": shadow_pixels,
        "shadow_fraction": shadow_pixels / mask.size,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 4. 预览固定映射到 0—255，保证不同实验的亮度可以直接比较。
    panels = [
        ("Background", background.astype(np.uint8)),
        ("Image", np.rint(image).astype(np.uint8)),
        ("Ground truth", mask * 255),
    ]
    scale = 2
    panel_width = IMAGE_WIDTH * scale
    panel_height = IMAGE_HEIGHT * scale
    title_height = 32
    gap = 12
    preview = Image.new(
        "RGB",
        (3 * panel_width + 4 * gap, panel_height + title_height + 2 * gap),
        "#20242b",
    )
    draw = ImageDraw.Draw(preview)
    for index, (title, pixels) in enumerate(panels):
        left = gap + index * (panel_width + gap)
        draw.text((left, gap + 6), title, fill="white")
        panel = Image.fromarray(pixels).resize(
            (panel_width, panel_height), Image.Resampling.NEAREST
        )
        preview.paste(panel, (left, gap + title_height))
    preview.save(output_dir / "preview.png")

    print(f"图像尺寸（高 × 宽） {IMAGE_HEIGHT} × {IMAGE_WIDTH}")
    print(f"暗区位置（x, y） ({SHADOW_X}, {SHADOW_Y})")
    print(f"暗区强度 {metadata['shadow_intensity']:g}")
    print(f"阴影像素数 {shadow_pixels}")
    print(f"阴影像素占比 {metadata['shadow_fraction']:.4%}")
    print(f"输出目录 {output_dir}")


if __name__ == "__main__":
    main()
