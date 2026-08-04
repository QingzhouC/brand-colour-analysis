from pathlib import Path
import colorsys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.cluster import KMeans
from skimage.color import lab2rgb, rgb2lab


# ============================================================
# 1. 基础设置
# ============================================================

# 存放图片的文件夹
IMAGE_FOLDER = Path("images")

# 输出结果的文件夹
OUTPUT_FOLDER = Path("results")

# 每张图片先提取多少个主要蓝色
COLOURS_PER_IMAGE = 10

# 最终从全部图片中提取多少种蓝色
FINAL_COLOUR_COUNT = 10

# 每张图片最多采样多少个蓝色像素
MAX_PIXELS_PER_IMAGE = 15000


# ============================================================
# 2. 蓝色像素过滤参数
# ============================================================

# HSV 明度低于这个值，认为太暗
MIN_VALUE = 0.25

# HSV 明度高于这个值，认为太接近白色或曝光高光
MAX_VALUE = 0.98

# 饱和度低于这个值，认为颜色太灰
# 这次图片本身偏高饱和，所以提高到 0.40
MIN_SATURATION = 0.40

# 只保留蓝色色相
#
# 180° 左右：青色
# 190°～210°：天蓝、海蓝
# 210°～230°：标准蓝
# 230°～250°：深蓝、偏紫蓝
#
# 设置为 190～250：
# 可以过滤掉大部分绿色和青绿色，
# 同时保留海蓝、天空蓝、钴蓝和深蓝。
BLUE_HUE_MIN = 190
BLUE_HUE_MAX = 250

# 支持的图片格式
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ============================================================
# 3. 工具函数
# ============================================================

def rgb_to_hex(rgb: np.ndarray) -> str:
    """
    RGB 转 HEX。

    示例：
    [16, 92, 244] -> #105CF4
    """

    rgb = np.clip(
        np.round(rgb),
        0,
        255
    ).astype(int)

    return "#{:02X}{:02X}{:02X}".format(
        rgb[0],
        rgb[1],
        rgb[2]
    )


def get_text_colour(rgb: np.ndarray) -> str:
    """
    根据色块亮度决定显示黑色文字还是白色文字。
    """

    rgb_normalised = (
        np.asarray(rgb, dtype=np.float32)
        / 255.0
    )

    red, green, blue = rgb_normalised

    luminance = (
        0.2126 * red
        + 0.7152 * green
        + 0.0722 * blue
    )

    return "black" if luminance > 0.55 else "white"


def calculate_hue(rgb: np.ndarray) -> float:
    """
    计算 HSV 色相角度，范围 0～360。
    """

    red, green, blue = (
        np.asarray(rgb, dtype=np.float32)
        / 255.0
    )

    hue, _, _ = colorsys.rgb_to_hsv(
        red,
        green,
        blue
    )

    return hue * 360


def calculate_saturation(rgb: np.ndarray) -> float:
    """
    计算 HSV 饱和度，范围 0～100。
    """

    red, green, blue = (
        np.asarray(rgb, dtype=np.float32)
        / 255.0
    )

    _, saturation, _ = colorsys.rgb_to_hsv(
        red,
        green,
        blue
    )

    return saturation * 100


def calculate_brightness(rgb: np.ndarray) -> float:
    """
    计算 HSV 明度，范围 0～100。
    """

    red, green, blue = (
        np.asarray(rgb, dtype=np.float32)
        / 255.0
    )

    _, _, brightness = colorsys.rgb_to_hsv(
        red,
        green,
        blue
    )

    return brightness * 100


def rgb_array_to_hsv_values(
    rgb_colours: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    批量将 RGB 数组转换为 HSV。

    输入：
    N × 3 的 RGB 数组，范围 0～255。

    返回：
    hue：0～360
    saturation：0～1
    value：0～1
    """

    rgb_normalised = (
        np.asarray(
            rgb_colours,
            dtype=np.float32
        )
        / 255.0
    )

    red = rgb_normalised[:, 0]
    green = rgb_normalised[:, 1]
    blue = rgb_normalised[:, 2]

    max_channel = np.max(
        rgb_normalised,
        axis=1
    )

    min_channel = np.min(
        rgb_normalised,
        axis=1
    )

    difference = (
        max_channel - min_channel
    )

    value = max_channel

    saturation = np.zeros_like(
        max_channel
    )

    non_zero_value = max_channel > 0

    saturation[non_zero_value] = (
        difference[non_zero_value]
        / max_channel[non_zero_value]
    )

    hue = np.zeros_like(
        max_channel
    )

    non_grey = difference > 0

    red_max = (
        non_grey
        & (max_channel == red)
    )

    green_max = (
        non_grey
        & (max_channel == green)
    )

    blue_max = (
        non_grey
        & (max_channel == blue)
    )

    hue[red_max] = (
        60
        * (
            (
                green[red_max]
                - blue[red_max]
            )
            / difference[red_max]
        )
    ) % 360

    hue[green_max] = (
        60
        * (
            (
                blue[green_max]
                - red[green_max]
            )
            / difference[green_max]
            + 2
        )
    )

    hue[blue_max] = (
        60
        * (
            (
                red[blue_max]
                - green[blue_max]
            )
            / difference[blue_max]
            + 4
        )
    )

    return hue, saturation, value


def is_valid_blue(rgb: np.ndarray) -> bool:
    """
    检查某个 RGB 颜色是否属于目标蓝色色域。
    """

    hue = calculate_hue(rgb)

    saturation = (
        calculate_saturation(rgb)
        / 100.0
    )

    brightness = (
        calculate_brightness(rgb)
        / 100.0
    )

    return (
        BLUE_HUE_MIN <= hue <= BLUE_HUE_MAX
        and saturation >= MIN_SATURATION
        and MIN_VALUE <= brightness <= MAX_VALUE
    )


# ============================================================
# 4. 读取图片并过滤像素
# ============================================================

def load_image_pixels(
    image_path: Path
) -> np.ndarray:
    """
    读取图片，只保留符合条件的蓝色像素。

    会过滤：

    1. 黑色和过暗像素；
    2. 白色和曝光高光；
    3. 灰色和低饱和颜色；
    4. 红色、橙色、黄色；
    5. 绿色和青绿色；
    6. 紫色和洋红色。
    """

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(
            image
        )

        image = image.convert("RGBA")

        # 透明图片统一铺在白色背景上
        white_background = Image.new(
            "RGBA",
            image.size,
            (255, 255, 255, 255)
        )

        image = Image.alpha_composite(
            white_background,
            image
        ).convert("RGB")

        # 缩小图片，降低计算量
        image.thumbnail(
            (500, 500)
        )

        pixels = np.asarray(
            image,
            dtype=np.float32
        ).reshape(-1, 3)

    if len(pixels) == 0:
        raise ValueError(
            "图片中没有可读取的像素。"
        )

    hue, saturation, value = (
        rgb_array_to_hsv_values(
            pixels
        )
    )

    # 只保留符合条件的高饱和蓝色像素
    blue_mask = (
        (hue >= BLUE_HUE_MIN)
        & (hue <= BLUE_HUE_MAX)
        & (saturation >= MIN_SATURATION)
        & (value >= MIN_VALUE)
        & (value <= MAX_VALUE)
    )

    filtered_pixels = pixels[
        blue_mask
    ]

    if len(filtered_pixels) == 0:
        raise ValueError(
            "图片中没有符合条件的高饱和蓝色像素。"
        )

    # 每张图片最多采样固定数量，避免大图影响过大
    if len(filtered_pixels) > MAX_PIXELS_PER_IMAGE:
        random_generator = (
            np.random.default_rng(42)
        )

        selected_indices = (
            random_generator.choice(
                len(filtered_pixels),
                size=MAX_PIXELS_PER_IMAGE,
                replace=False
            )
        )

        filtered_pixels = (
            filtered_pixels[
                selected_indices
            ]
        )

    return filtered_pixels


# ============================================================
# 5. 从单张图片提取主要蓝色
# ============================================================

def extract_image_colours(
    image_path: Path,
    colour_count: int = COLOURS_PER_IMAGE
) -> list[dict]:
    """
    在 Lab 色彩空间中对单张图片的蓝色像素聚类。
    """

    rgb_pixels = load_image_pixels(
        image_path
    )

    rgb_normalised = (
        rgb_pixels / 255.0
    )

    lab_pixels = rgb2lab(
        rgb_normalised.reshape(
            -1,
            1,
            3
        )
    ).reshape(-1, 3)

    # 计算实际存在多少种不同颜色
    unique_colour_count = len(
        np.unique(
            np.round(
                rgb_pixels
            ).astype(np.uint8),
            axis=0
        )
    )

    actual_cluster_count = min(
        colour_count,
        unique_colour_count,
        len(lab_pixels)
    )

    if actual_cluster_count < 1:
        raise ValueError(
            "图片无法提取蓝色。"
        )

    model = KMeans(
        n_clusters=actual_cluster_count,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        lab_pixels
    )

    counts = np.bincount(
        labels,
        minlength=actual_cluster_count
    )

    proportions = (
        counts / counts.sum()
    )

    cluster_lab = (
        model.cluster_centers_
    )

    cluster_rgb = lab2rgb(
        cluster_lab.reshape(
            1,
            -1,
            3
        )
    ).reshape(-1, 3)

    cluster_rgb = np.clip(
        cluster_rgb * 255,
        0,
        255
    )

    results = []

    for index in range(
        actual_cluster_count
    ):
        rgb = cluster_rgb[index]
        lab = cluster_lab[index]

        # 再检查一次聚类中心，防止 Lab 转 RGB 后偏出蓝色色域
        if not is_valid_blue(rgb):
            continue

        results.append({
            "rgb": rgb,
            "lab": lab,
            "hex": rgb_to_hex(rgb),
            "proportion": float(
                proportions[index]
            ),
        })

    results.sort(
        key=lambda item: item["proportion"],
        reverse=True
    )

    if not results:
        raise ValueError(
            "聚类后没有得到符合要求的蓝色。"
        )

    return results


# ============================================================
# 6. 分析全部图片
# ============================================================

def analyse_all_images(
) -> tuple[pd.DataFrame, list[dict]]:
    """
    遍历 images 文件夹中的所有图片，
    提取每张图片中的主要蓝色。
    """

    if not IMAGE_FOLDER.exists():
        raise FileNotFoundError(
            "没有找到 images 文件夹。"
        )

    image_files = sorted([
        file_path
        for file_path in IMAGE_FOLDER.iterdir()
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])

    if not image_files:
        raise FileNotFoundError(
            "images 文件夹中没有找到支持的图片。"
        )

    print(
        f"找到 {len(image_files)} 张图片。"
    )

    print(
        "开始分析图片中的高饱和蓝色……"
    )

    csv_rows = []
    all_colour_samples = []

    successful_image_count = 0
    skipped_image_count = 0

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
                f"跳过 {image_path.name}，"
                f"原因：{error}"
            )

            skipped_image_count += 1
            continue

        successful_image_count += 1

        for rank, colour in enumerate(
            image_colours,
            start=1
        ):
            rgb = colour["rgb"]
            lab = colour["lab"]
            proportion = colour["proportion"]

            csv_rows.append({
                "image_name": image_path.name,
                "colour_rank": rank,
                "hex": colour["hex"],
                "red": round(float(rgb[0])),
                "green": round(float(rgb[1])),
                "blue": round(float(rgb[2])),
                "hue": round(
                    calculate_hue(rgb),
                    2
                ),
                "saturation": round(
                    calculate_saturation(rgb),
                    2
                ),
                "brightness": round(
                    calculate_brightness(rgb),
                    2
                ),
                "proportion_percent": round(
                    proportion * 100,
                    2
                ),
                "lab_l": round(
                    float(lab[0]),
                    2
                ),
                "lab_a": round(
                    float(lab[1]),
                    2
                ),
                "lab_b": round(
                    float(lab[2]),
                    2
                ),
            })

            # 保存给第二轮总聚类
            all_colour_samples.append({
                "lab": np.asarray(
                    lab,
                    dtype=np.float32
                ),
                "rgb": np.asarray(
                    rgb,
                    dtype=np.float32
                ),
                "weight": float(
                    proportion
                ),
                "image_name": image_path.name,
            })

    if not csv_rows:
        raise RuntimeError(
            "没有任何图片成功完成分析。"
        )

    print()
    print(
        f"成功分析：{successful_image_count} 张"
    )

    print(
        f"跳过图片：{skipped_image_count} 张"
    )

    image_colour_dataframe = (
        pd.DataFrame(csv_rows)
    )

    return (
        image_colour_dataframe,
        all_colour_samples
    )


# ============================================================
# 7. 从全部图片中最终提取 10 个蓝色
# ============================================================

def create_final_colour_clusters(
    colour_samples: list[dict]
) -> pd.DataFrame:
    """
    从所有图片的颜色样本中聚类出 10 个蓝色。

    这一版：

    1. 不加入固定品牌色；
    2. 不保留非蓝色；
    3. 最终全部 10 个颜色都来自图片；
    4. 使用每个颜色在原图中的占比作为聚类权重。
    """

    if not colour_samples:
        raise ValueError(
            "没有可用于最终聚类的颜色样本。"
        )

    lab_colours = np.array(
        [
            sample["lab"]
            for sample in colour_samples
        ],
        dtype=np.float32
    )

    rgb_colours = np.array(
        [
            sample["rgb"]
            for sample in colour_samples
        ],
        dtype=np.float32
    )

    sample_weights = np.array(
        [
            sample["weight"]
            for sample in colour_samples
        ],
        dtype=np.float64
    )

    hue, saturation, value = (
        rgb_array_to_hsv_values(
            rgb_colours
        )
    )

    # 最终聚类前再过滤一次
    blue_mask = (
        (hue >= BLUE_HUE_MIN)
        & (hue <= BLUE_HUE_MAX)
        & (saturation >= MIN_SATURATION)
        & (value >= MIN_VALUE)
        & (value <= MAX_VALUE)
    )

    lab_colours = lab_colours[
        blue_mask
    ]

    rgb_colours = rgb_colours[
        blue_mask
    ]

    sample_weights = sample_weights[
        blue_mask
    ]

    if len(lab_colours) == 0:
        raise ValueError(
            "过滤后没有可以用于聚类的蓝色样本。"
        )

    actual_cluster_count = min(
        FINAL_COLOUR_COUNT,
        len(lab_colours)
    )

    model = KMeans(
        n_clusters=actual_cluster_count,
        random_state=42,
        n_init=30
    )

    # 使用权重进行聚类
    model.fit(
        lab_colours,
        sample_weight=sample_weights
    )

    labels = model.labels_

    cluster_centres = (
        model.cluster_centers_
    )

    rows = []

    total_weight = float(
        sample_weights.sum()
    )

    for cluster_index in range(
        actual_cluster_count
    ):
        member_indices = np.where(
            labels == cluster_index
        )[0]

        if len(member_indices) == 0:
            continue

        member_lab = lab_colours[
            member_indices
        ]

        member_rgb = rgb_colours[
            member_indices
        ]

        member_weights = sample_weights[
            member_indices
        ]

        centre_lab = cluster_centres[
            cluster_index
        ]

        # 找出最接近聚类中心的真实颜色样本
        distances = np.linalg.norm(
            member_lab - centre_lab,
            axis=1
        )

        closest_position = np.argmin(
            distances
        )

        representative_rgb = member_rgb[
            closest_position
        ]

        representative_lab = member_lab[
            closest_position
        ]

        cluster_weight = float(
            member_weights.sum()
        )

        cluster_percentage = (
            cluster_weight
            / total_weight
            * 100
        )

        # 最后再次确保结果属于蓝色
        if not is_valid_blue(
            representative_rgb
        ):
            continue

        rows.append({
            "hex": rgb_to_hex(
                representative_rgb
            ),
            "red": round(
                float(
                    representative_rgb[0]
                )
            ),
            "green": round(
                float(
                    representative_rgb[1]
                )
            ),
            "blue": round(
                float(
                    representative_rgb[2]
                )
            ),
            "hue": round(
                calculate_hue(
                    representative_rgb
                ),
                2
            ),
            "saturation": round(
                calculate_saturation(
                    representative_rgb
                ),
                2
            ),
            "brightness": round(
                calculate_brightness(
                    representative_rgb
                ),
                2
            ),
            "frequency_percent": round(
                cluster_percentage,
                2
            ),
            "lab_l": round(
                float(
                    representative_lab[0]
                ),
                2
            ),
            "lab_a": round(
                float(
                    representative_lab[1]
                ),
                2
            ),
            "lab_b": round(
                float(
                    representative_lab[2]
                ),
                2
            ),
        })

    dataframe = pd.DataFrame(
        rows
    )

    if dataframe.empty:
        raise ValueError(
            "最终没有得到符合条件的蓝色。"
        )

    # 先按照色相排列：
    # 偏青蓝 -> 标准蓝 -> 偏紫蓝
    dataframe = dataframe.sort_values(
        by=[
            "hue",
            "brightness"
        ],
        ascending=[
            True,
            False
        ]
    ).reset_index(drop=True)

    dataframe.insert(
        0,
        "rank",
        range(
            1,
            len(dataframe) + 1
        )
    )

    return dataframe


# ============================================================
# 8. 生成横向色板
# ============================================================

def create_palette_image(
    final_colours: pd.DataFrame
) -> None:
    """
    生成横向的 10 个蓝色色板。
    """

    colour_count = len(
        final_colours
    )

    figure, axis = plt.subplots(
        figsize=(18, 5)
    )

    for position, (_, row) in enumerate(
        final_colours.iterrows()
    ):
        rgb = np.array(
            [
                row["red"],
                row["green"],
                row["blue"],
            ],
            dtype=np.float32
        )

        axis.barh(
            y=0,
            width=1,
            left=position,
            height=1,
            color=rgb / 255.0
        )

        axis.text(
            position + 0.5,
            0,
            (
                f"{int(row['rank']):02d}\n"
                f"{row['hex']}\n"
                f"H {row['hue']:.0f}°\n"
                f"S {row['saturation']:.0f}%"
            ),
            horizontalalignment="center",
            verticalalignment="center",
            color=get_text_colour(rgb),
            fontsize=9,
            fontweight="bold"
        )

    axis.set_xlim(
        0,
        colour_count
    )

    axis.set_ylim(
        -0.6,
        0.6
    )

    axis.axis("off")

    axis.set_title(
        (
            "Top 10 High-Saturation Blues\n"
            "Only blue pixels retained"
        ),
        fontsize=18,
        pad=20
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER
        / "top_10_blue_colours.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"已生成蓝色色板：{output_path}"
    )


# ============================================================
# 9. 生成独立色卡
# ============================================================

def create_colour_cards(
    final_colours: pd.DataFrame
) -> None:
    """
    生成 10 个蓝色的独立色卡。
    """

    colour_count = len(
        final_colours
    )

    figure, axes = plt.subplots(
        nrows=colour_count,
        ncols=1,
        figsize=(
            10,
            max(
                3,
                colour_count * 1.25
            )
        )
    )

    if colour_count == 1:
        axes = [axes]

    for axis, (_, row) in zip(
        axes,
        final_colours.iterrows()
    ):
        rgb = np.array(
            [
                row["red"],
                row["green"],
                row["blue"],
            ],
            dtype=np.float32
        )

        axis.set_facecolor(
            rgb / 255.0
        )

        axis.text(
            0.03,
            0.5,
            (
                f"{int(row['rank']):02d}   "
                f"{row['hex']}   "
                f"RGB("
                f"{int(row['red'])}, "
                f"{int(row['green'])}, "
                f"{int(row['blue'])})   "
                f"H {row['hue']:.1f}°   "
                f"S {row['saturation']:.1f}%   "
                f"V {row['brightness']:.1f}%   "
                f"Weight {row['frequency_percent']:.1f}%"
            ),
            transform=axis.transAxes,
            verticalalignment="center",
            color=get_text_colour(rgb),
            fontsize=11,
            fontweight="bold"
        )

        axis.set_xticks([])
        axis.set_yticks([])

        for spine in axis.spines.values():
            spine.set_visible(False)

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER
        / "top_10_blue_colour_cards.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"已生成蓝色色卡：{output_path}"
    )


# ============================================================
# 10. 生成按出现频率排列的色板
# ============================================================

def create_frequency_palette(
    final_colours: pd.DataFrame
) -> None:
    """
    根据颜色在图片中的出现权重生成比例色板。

    色块越宽，代表这个蓝色在图片中越常见。
    """

    frequency_colours = (
        final_colours.sort_values(
            by="frequency_percent",
            ascending=False
        ).reset_index(drop=True)
    )

    frequencies = (
        frequency_colours[
            "frequency_percent"
        ].to_numpy(
            dtype=np.float64
        )
    )

    frequency_sum = frequencies.sum()

    if frequency_sum <= 0:
        frequencies = np.ones(
            len(frequency_colours)
        )

        frequency_sum = frequencies.sum()

    widths = (
        frequencies / frequency_sum
    )

    figure, axis = plt.subplots(
        figsize=(18, 5)
    )

    current_left = 0.0

    for _, row in (
        frequency_colours.iterrows()
    ):
        rgb = np.array(
            [
                row["red"],
                row["green"],
                row["blue"],
            ],
            dtype=np.float32
        )

        width = (
            float(
                row["frequency_percent"]
            )
            / frequency_sum
        )

        axis.barh(
            y=0,
            width=width,
            left=current_left,
            height=1,
            color=rgb / 255.0
        )

        # 色块太窄时不放文字，避免重叠
        if width >= 0.065:
            axis.text(
                current_left + width / 2,
                0,
                (
                    f"{row['hex']}\n"
                    f"{row['frequency_percent']:.1f}%"
                ),
                horizontalalignment="center",
                verticalalignment="center",
                color=get_text_colour(rgb),
                fontsize=9,
                fontweight="bold"
            )

        current_left += width

    axis.set_xlim(
        0,
        1
    )

    axis.set_ylim(
        -0.6,
        0.6
    )

    axis.axis("off")

    axis.set_title(
        (
            "Blue Colour Frequency Distribution\n"
            "Wider blocks represent more frequent colours"
        ),
        fontsize=18,
        pad=20
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER
        / "blue_colour_frequency_palette.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"已生成蓝色频率色板：{output_path}"
    )


# ============================================================
# 11. 主程序
# ============================================================

def main() -> None:
    """
    执行完整蓝色分析流程。
    """

    if not IMAGE_FOLDER.exists():
        IMAGE_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        raise FileNotFoundError(
            "没有找到 images 文件夹。"
            "系统已经自动创建，请把图片放进去后重新运行。"
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 第一步：分析每张图片
    # --------------------------------------------------------

    (
        image_colours,
        all_colour_samples
    ) = analyse_all_images()

    image_colours_path = (
        OUTPUT_FOLDER
        / "blue_colours_from_each_image.csv"
    )

    image_colours.to_csv(
        image_colours_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        "每张图片的蓝色数据已保存："
        f"{image_colours_path}"
    )

    # --------------------------------------------------------
    # 第二步：从全部图片中聚类出 10 个蓝色
    # --------------------------------------------------------

    final_colours = (
        create_final_colour_clusters(
            all_colour_samples
        )
    )

    final_colours_path = (
        OUTPUT_FOLDER
        / "top_10_blue_colour_candidates.csv"
    )

    final_colours.to_csv(
        final_colours_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "最终蓝色结果已保存："
        f"{final_colours_path}"
    )

    # --------------------------------------------------------
    # 第三步：生成可视化图片
    # --------------------------------------------------------

    create_palette_image(
        final_colours
    )

    create_colour_cards(
        final_colours
    )

    create_frequency_palette(
        final_colours
    )

    # --------------------------------------------------------
    # 第四步：终端打印结果
    # --------------------------------------------------------

    print()
    print("分析完成。")
    print()

    print(
        "以下为图片中提取出的 10 个高饱和蓝色："
    )

    for _, row in final_colours.iterrows():
        print(
            f"{int(row['rank']):02d}. "
            f"{row['hex']}  "
            f"RGB("
            f"{int(row['red'])}, "
            f"{int(row['green'])}, "
            f"{int(row['blue'])})  "
            f"H={row['hue']:.1f}°  "
            f"S={row['saturation']:.1f}%  "
            f"V={row['brightness']:.1f}%  "
            f"Weight={row['frequency_percent']:.1f}%"
        )


if __name__ == "__main__":
    main()