from pathlib import Path
import colorsys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans
from skimage.color import lab2rgb, rgb2lab


# ============================================================
# 1. 基础设置
# ============================================================

# 存放图片的文件夹
IMAGE_FOLDER = Path("images")

# 输出结果的文件夹
OUTPUT_FOLDER = Path("results")

# 每张图片提取多少个主要颜色
COLOURS_PER_IMAGE = 5

# 最后把所有图片的颜色归纳成多少个代表色
FINAL_COLOUR_COUNT = 10

# 每张图片最多采样多少个像素
# 数值越大越准确，但运行越慢
MAX_PIXELS_PER_IMAGE = 15000

# 支持的图片格式
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# 2. 工具函数
# ============================================================

def rgb_to_hex(rgb: np.ndarray) -> str:
    """
    把 RGB 数值转换为 HEX。
    例如 [27, 88, 244] 变成 #1B58F4。
    """
    rgb = np.clip(np.round(rgb), 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])


def hex_to_rgb(hex_colour: str) -> tuple[int, int, int]:
    """
    把 HEX 转换为 RGB。
    """
    hex_colour = hex_colour.lstrip("#")
    return tuple(
        int(hex_colour[i:i + 2], 16)
        for i in (0, 2, 4)
    )


def get_text_colour(rgb: np.ndarray) -> str:
    """
    判断色块上应该使用黑色文字还是白色文字。
    这样输出的色板比较容易阅读。
    """
    r, g, b = rgb / 255

    luminance = (
        0.2126 * r
        + 0.7152 * g
        + 0.0722 * b
    )

    return "black" if luminance > 0.55 else "white"


def calculate_hue(rgb: np.ndarray) -> float:
    """
    计算颜色的 Hue 色相角度。
    返回范围为 0 到 360。
    """
    r, g, b = rgb / 255
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return h * 360


def calculate_saturation(rgb: np.ndarray) -> float:
    """
    计算 HSV 饱和度。
    返回范围为 0 到 100。
    """
    r, g, b = rgb / 255
    _, s, _ = colorsys.rgb_to_hsv(r, g, b)
    return s * 100


def calculate_brightness(rgb: np.ndarray) -> float:
    """
    计算 HSV 明度。
    返回范围为 0 到 100。
    """
    r, g, b = rgb / 255
    _, _, v = colorsys.rgb_to_hsv(r, g, b)
    return v * 100


# ============================================================
# 3. 读取图片
# ============================================================

def load_image_pixels(image_path: Path) -> np.ndarray:
    """
    读取图片，并转换成 RGB 像素数组。

    处理内容：
    1. 自动修正图片方向；
    2. 去掉透明背景；
    3. 缩小图片，减少运算量；
    4. 随机抽样像素。
    """

    with Image.open(image_path) as image:
        # 统一转换为 RGBA，方便处理透明图片
        image = image.convert("RGBA")

        # 创建白色背景
        white_background = Image.new(
            "RGBA",
            image.size,
            (255, 255, 255, 255)
        )

        # 把透明图片放到白色背景上
        image = Image.alpha_composite(
            white_background,
            image
        ).convert("RGB")

        # 缩小图片，最长边不超过 500px
        image.thumbnail((500, 500))

        # 转换成 NumPy 数组
        pixels = np.asarray(image, dtype=np.float32)

        # 从 高×宽×3 变成 像素数量×3
        pixels = pixels.reshape(-1, 3)

        # 删除接近纯白色的像素
        # 避免图片白边和白色背景影响结果
        brightness = pixels.mean(axis=1)
        pixels = pixels[brightness < 248]

        if len(pixels) == 0:
            raise ValueError("图片中没有可分析的有效像素。")

        # 如果像素太多，随机抽样
        if len(pixels) > MAX_PIXELS_PER_IMAGE:
            random_generator = np.random.default_rng(42)

            selected_indices = random_generator.choice(
                len(pixels),
                size=MAX_PIXELS_PER_IMAGE,
                replace=False
            )

            pixels = pixels[selected_indices]

        return pixels


# ============================================================
# 4. 从单张图片提取主要颜色
# ============================================================

def extract_image_colours(
    image_path: Path,
    colour_count: int = 5
) -> list[dict]:
    """
    对一张图片进行颜色聚类。

    不是直接在 RGB 中计算，
    而是先转换到 Lab 色彩空间。

    Lab 更接近人眼对颜色差异的感受。
    """

    rgb_pixels = load_image_pixels(image_path)

    # RGB 需要先转换到 0 到 1
    rgb_normalised = rgb_pixels / 255

    # 转换到 Lab 色彩空间
    lab_pixels = rgb2lab(
        rgb_normalised.reshape(-1, 1, 3)
    ).reshape(-1, 3)

    # 图片颜色很少时，防止聚类数量超过像素种类
    unique_colour_count = len(
        np.unique(
            np.round(rgb_pixels).astype(int),
            axis=0
        )
    )

    actual_cluster_count = min(
        colour_count,
        unique_colour_count,
        len(lab_pixels)
    )

    if actual_cluster_count < 1:
        raise ValueError("图片无法提取颜色。")

    # K-means 聚类
    model = KMeans(
        n_clusters=actual_cluster_count,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(lab_pixels)

    # 每个颜色组包含多少像素
    counts = np.bincount(
        labels,
        minlength=actual_cluster_count
    )

    proportions = counts / counts.sum()

    # Lab聚类中心转换回RGB
    cluster_lab = model.cluster_centers_

    cluster_rgb = lab2rgb(
        cluster_lab.reshape(1, -1, 3)
    ).reshape(-1, 3)

    cluster_rgb = np.clip(
        cluster_rgb * 255,
        0,
        255
    )

    results = []

    for index in range(actual_cluster_count):
        rgb = cluster_rgb[index]

        results.append({
            "rgb": rgb,
            "lab": cluster_lab[index],
            "hex": rgb_to_hex(rgb),
            "proportion": float(proportions[index]),
        })

    # 按照颜色在图片中的占比排序
    results.sort(
        key=lambda item: item["proportion"],
        reverse=True
    )

    return results


# ============================================================
# 5. 分析所有图片
# ============================================================

def analyse_all_images() -> tuple[pd.DataFrame, list[dict]]:
    """
    遍历 images 文件夹中的所有图片。
    每张图提取指定数量的主要颜色。
    """

    image_files = sorted([
        file_path
        for file_path in IMAGE_FOLDER.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        raise FileNotFoundError(
            "images 文件夹中没有找到图片。"
        )

    print(f"找到 {len(image_files)} 张图片。")
    print("开始分析图片颜色……")

    csv_rows = []
    all_colour_samples = []

    for image_number, image_path in enumerate(
        image_files,
        start=1
    ):
        print(
            f"[{image_number}/{len(image_files)}] "
            f"正在分析：{image_path.name}"
        )

        try:
            image_colours = extract_image_colours(
                image_path,
                COLOURS_PER_IMAGE
            )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ) as error:
            print(
                f"跳过 {image_path.name}，原因：{error}"
            )
            continue

        for rank, colour in enumerate(
            image_colours,
            start=1
        ):
            rgb = colour["rgb"]
            lab = colour["lab"]

            csv_rows.append({
                "image_name": image_path.name,
                "colour_rank": rank,
                "hex": colour["hex"],
                "red": round(float(rgb[0])),
                "green": round(float(rgb[1])),
                "blue": round(float(rgb[2])),
                "lab_l": round(float(lab[0]), 2),
                "lab_a": round(float(lab[1]), 2),
                "lab_b": round(float(lab[2]), 2),
                "image_proportion_percent": round(
                    colour["proportion"] * 100,
                    2
                ),
            })

            # 保存给第二次总聚类
            all_colour_samples.append({
                "lab": colour["lab"],
                "rgb": colour["rgb"],
                "weight": colour["proportion"],
                "image_name": image_path.name,
            })

    if not csv_rows:
        raise RuntimeError(
            "没有任何图片成功完成分析。"
        )

    image_colour_dataframe = pd.DataFrame(csv_rows)

    return image_colour_dataframe, all_colour_samples


# ============================================================
# 6. 对所有图片颜色进行第二次聚类
# ============================================================

def create_final_colour_clusters(
    colour_samples: list[dict]
) -> pd.DataFrame:
    """
    将所有图片提取出的颜色再次聚类。

    例如：
    100张图片 × 每张5种颜色
    = 500个颜色样本

    然后把500个颜色归纳成10个最终代表色。
    """

    lab_colours = np.array([
        sample["lab"]
        for sample in colour_samples
    ])

    sample_weights = np.array([
        sample["weight"]
        for sample in colour_samples
    ])

    cluster_count = min(
        FINAL_COLOUR_COUNT,
        len(lab_colours)
    )

    model = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=20
    )

    # sample_weight 能让图片中占比更大的颜色更重要
    labels = model.fit_predict(
        lab_colours,
        sample_weight=sample_weights
    )

    cluster_lab = model.cluster_centers_

    # 转回 RGB
    cluster_rgb = lab2rgb(
        cluster_lab.reshape(1, -1, 3)
    ).reshape(-1, 3)

    cluster_rgb = np.clip(
        cluster_rgb * 255,
        0,
        255
    )

    # 计算每一个最终颜色的总权重
    cluster_weights = np.zeros(cluster_count)

    for sample_index, cluster_index in enumerate(labels):
        cluster_weights[cluster_index] += (
            sample_weights[sample_index]
        )

    cluster_percentages = (
        cluster_weights /
        cluster_weights.sum() *
        100
    )

    rows = []

    for cluster_index in range(cluster_count):
        rgb = cluster_rgb[cluster_index]
        lab = cluster_lab[cluster_index]

        rows.append({
            "cluster": cluster_index + 1,
            "hex": rgb_to_hex(rgb),
            "red": round(float(rgb[0])),
            "green": round(float(rgb[1])),
            "blue": round(float(rgb[2])),
            "lab_l": round(float(lab[0]), 2),
            "lab_a": round(float(lab[1]), 2),
            "lab_b": round(float(lab[2]), 2),
            "hue": round(calculate_hue(rgb), 2),
            "saturation": round(
                calculate_saturation(rgb),
                2
            ),
            "brightness": round(
                calculate_brightness(rgb),
                2
            ),
            "percentage": round(
                float(cluster_percentages[cluster_index]),
                2
            ),
        })

    dataframe = pd.DataFrame(rows)

    # 按照颜色占比从高到低排列
    dataframe = dataframe.sort_values(
        by="percentage",
        ascending=False
    ).reset_index(drop=True)

    # 重新编号
    dataframe["cluster"] = range(
        1,
        len(dataframe) + 1
    )

    return dataframe


# ============================================================
# 7. 生成最终色板图片
# ============================================================

def create_palette_image(
    final_colours: pd.DataFrame
) -> None:
    """
    创建一张横向颜色条图片。
    每一块的宽度代表颜色出现的比例。
    """

    figure, axis = plt.subplots(
        figsize=(16, 5)
    )

    current_x = 0

    for _, row in final_colours.iterrows():
        rgb = np.array([
            row["red"],
            row["green"],
            row["blue"],
        ])

        width = row["percentage"]

        axis.barh(
            y=0,
            width=width,
            left=current_x,
            height=1,
            color=rgb / 255
        )

        # 色块足够宽时才显示文字
        if width >= 5:
            axis.text(
                current_x + width / 2,
                0,
                f"{row['hex']}\n{row['percentage']}%",
                horizontalalignment="center",
                verticalalignment="center",
                color=get_text_colour(rgb),
                fontsize=11,
                fontweight="bold"
            )

        current_x += width

    axis.set_xlim(0, 100)
    axis.set_ylim(-0.6, 0.6)
    axis.axis("off")

    axis.set_title(
        "Final Colour Clusters",
        fontsize=20,
        pad=20
    )

    plt.tight_layout()

    output_path = OUTPUT_FOLDER / "final_colour_palette.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"已生成色板：{output_path}")


def create_colour_cards(
    final_colours: pd.DataFrame
) -> None:
    """
    创建独立色卡。
    每个颜色以相同面积显示，方便观察。
    """

    colour_count = len(final_colours)

    figure, axes = plt.subplots(
        nrows=colour_count,
        ncols=1,
        figsize=(8, colour_count * 1.2)
    )

    if colour_count == 1:
        axes = [axes]

    for axis, (_, row) in zip(
        axes,
        final_colours.iterrows()
    ):
        rgb = np.array([
            row["red"],
            row["green"],
            row["blue"],
        ])

        axis.set_facecolor(rgb / 255)

        axis.text(
            0.03,
            0.5,
            (
                f"{row['hex']}    "
                f"{row['percentage']}%    "
                f"RGB({row['red']}, "
                f"{row['green']}, "
                f"{row['blue']})"
            ),
            transform=axis.transAxes,
            verticalalignment="center",
            color=get_text_colour(rgb),
            fontsize=13,
            fontweight="bold"
        )

        axis.set_xticks([])
        axis.set_yticks([])

        for spine in axis.spines.values():
            spine.set_visible(False)

    plt.tight_layout()

    output_path = OUTPUT_FOLDER / "final_colour_cards.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"已生成色卡：{output_path}")


# ============================================================
# 8. 主程序
# ============================================================

def main() -> None:
    """
    执行完整分析流程。
    """

    if not IMAGE_FOLDER.exists():
        IMAGE_FOLDER.mkdir(parents=True)

        raise FileNotFoundError(
            "没有找到 images 文件夹。"
            "系统已自动创建，请把图片放进去后重新运行。"
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # 第一次：分析每一张图片
    image_colours, all_colour_samples = (
        analyse_all_images()
    )

    image_colours_path = (
        OUTPUT_FOLDER /
        "colours_from_each_image.csv"
    )

    image_colours.to_csv(
        image_colours_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"每张图片的颜色数据已保存："
        f"{image_colours_path}"
    )

    # 第二次：归纳所有图片的颜色
    final_colours = create_final_colour_clusters(
        all_colour_samples
    )

    final_colours_path = (
        OUTPUT_FOLDER /
        "final_colour_clusters.csv"
    )

    final_colours.to_csv(
        final_colours_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"最终颜色结果已保存："
        f"{final_colours_path}"
    )

    # 输出色板
    create_palette_image(final_colours)
    create_colour_cards(final_colours)

    print()
    print("分析完成。")
    print()
    print("最终代表颜色：")

    for _, row in final_colours.iterrows():
        print(
            f"{row['cluster']:02d}. "
            f"{row['hex']} "
            f"占比 {row['percentage']}%"
        )


if __name__ == "__main__":
    main()