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

# 每张图片提取多少个主要颜色
COLOURS_PER_IMAGE = 10

# 最终输出多少种颜色
# 包含固定目标品牌色 #105CF4
FINAL_COLOUR_COUNT = 10

# 每张图片最多采样多少个像素
MAX_PIXELS_PER_IMAGE = 15000

# 图片像素过滤参数
# HSV Value 小于该值时视为过暗
MIN_VALUE = 0.22

# HSV Value 大于该值时视为接近纯白
MAX_VALUE = 0.98

# 饱和度低于该值时视为接近灰色
MIN_SATURATION = 0.18

# 只分析蓝色和青蓝色

BLUE_HUE_MIN = 180

BLUE_HUE_MAX = 245

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
    把 RGB 转换为 HEX。

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




def get_text_colour(
    rgb: np.ndarray
) -> str:
    """
    判断色块文字应该显示为黑色还是白色。
    """

    rgb_normalised = (
        np.asarray(rgb, dtype=np.float32) / 255.0
    )

    red, green, blue = rgb_normalised

    luminance = (
        0.2126 * red
        + 0.7152 * green
        + 0.0722 * blue
    )

    return "black" if luminance > 0.55 else "white"


def calculate_hue(
    rgb: np.ndarray
) -> float:
    """
    计算 HSV 色相角度，范围 0～360。
    """

    red, green, blue = (
        np.asarray(rgb, dtype=np.float32) / 255.0
    )

    hue, _, _ = colorsys.rgb_to_hsv(
        red,
        green,
        blue
    )

    return hue * 360


def calculate_saturation(
    rgb: np.ndarray
) -> float:
    """
    计算 HSV 饱和度，范围 0～100。
    """

    red, green, blue = (
        np.asarray(rgb, dtype=np.float32) / 255.0
    )

    _, saturation, _ = colorsys.rgb_to_hsv(
        red,
        green,
        blue
    )

    return saturation * 100


def calculate_brightness(
    rgb: np.ndarray
) -> float:
    """
    计算 HSV 明度，范围 0～100。
    """

    red, green, blue = (
        np.asarray(rgb, dtype=np.float32) / 255.0
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
    批量计算 RGB 颜色的 HSV。

    输入：
    N × 3 的 RGB 数组，范围 0～255。

    返回：
    hue：0～360
    saturation：0～1
    value：0～1
    """

    rgb_normalised = (
        np.asarray(rgb_colours, dtype=np.float32)
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


# ============================================================
# 3. 读取和过滤图片像素
# ============================================================

def load_image_pixels(
    image_path: Path
) -> np.ndarray:
    """
    读取海洋图片，并且只保留蓝色色域像素。

    会过滤：
    1. 黑色和过暗像素；
    2. 白色高光；
    3. 灰色和低饱和颜色；
    4. 棕色、绿色、红色等非蓝色像素。
    """

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGBA")

        white_background = Image.new(
            "RGBA",
            image.size,
            (255, 255, 255, 255)
        )

        image = Image.alpha_composite(
            white_background,
            image
        ).convert("RGB")

        image.thumbnail((500, 500))

        pixels = np.asarray(
            image,
            dtype=np.float32
        ).reshape(-1, 3)

    if len(pixels) == 0:
        raise ValueError(
            "图片中没有可读取的像素。"
        )

    hue, saturation, value = (
        rgb_array_to_hsv_values(pixels)
    )

    # 只保留有效蓝色
    valid_colour_mask = (
        (value > MIN_VALUE)
        & (value < MAX_VALUE)
        & (saturation > MIN_SATURATION)
        & (hue >= BLUE_HUE_MIN)
        & (hue <= BLUE_HUE_MAX)
    )

    filtered_pixels = pixels[
        valid_colour_mask
    ]

    if len(filtered_pixels) == 0:
        raise ValueError(
            "图片中没有符合条件的蓝色像素。"
        )

    if (
        len(filtered_pixels)
        > MAX_PIXELS_PER_IMAGE
    ):
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

        filtered_pixels = filtered_pixels[
            selected_indices
        ]

    return filtered_pixels


# ============================================================
# 4. 单张图片颜色提取
# ============================================================

def extract_image_colours(
    image_path: Path,
    colour_count: int = COLOURS_PER_IMAGE
) -> list[dict]:
    """
    对单张图片进行 Lab 空间 K-Means 聚类。
    """

    rgb_pixels = load_image_pixels(
        image_path
    )

    rgb_normalised = (
        rgb_pixels / 255.0
    )

    lab_pixels = rgb2lab(
        rgb_normalised.reshape(-1, 1, 3)
    ).reshape(-1, 3)

    # 防止聚类数大于实际颜色数
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
            "图片无法提取颜色。"
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
        cluster_lab.reshape(1, -1, 3)
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

    return results


# ============================================================
# 5. 分析所有图片
# ============================================================

def analyse_all_images(
) -> tuple[pd.DataFrame, list[dict]]:
    """
    遍历 images 文件夹中的所有图片，
    并提取每张图片的主要颜色。
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
    print("开始分析图片颜色……")

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
            image_colours = (
                extract_image_colours(
                    image_path,
                    COLOURS_PER_IMAGE
                )
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
            proportion = colour[
                "proportion"
            ]

            csv_rows.append({
                "image_name": image_path.name,
                "colour_rank": rank,
                "hex": colour["hex"],
                "red": round(float(rgb[0])),
                "green": round(float(rgb[1])),
                "blue": round(float(rgb[2])),
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

            # 保存给第二轮聚类
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
                "image_name":
                    image_path.name,
            })

    if not csv_rows:
        raise RuntimeError(
            "没有任何图片成功完成分析。"
        )

    print()
    print(
        f"成功分析："
        f"{successful_image_count} 张"
    )
    print(
        f"跳过图片："
        f"{skipped_image_count} 张"
    )

    image_colour_dataframe = (
        pd.DataFrame(csv_rows)
    )

    return (
        image_colour_dataframe,
        all_colour_samples
    )


# ============================================================
# 6. 目标导向的最终聚类
# ============================================================

def create_final_colour_clusters(
    colour_samples: list[dict]
) -> pd.DataFrame:
    """
    从全部海洋图片中提取 9 个高频蓝色，
    再加入 #105CF4 作为品牌方向候选色。

    最终不显示百分比，只输出 10 个候选色。
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

    # 只保留蓝色色域，同时过滤过暗、过白和低饱和颜色
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
            "过滤后没有可用于聚类的蓝色样本。"
        )

    # 这里只提取 9 个图片高频蓝色
    extracted_colour_count = min(
        FINAL_COLOUR_COUNT - 1,
        len(lab_colours)
    )

    model = KMeans(
        n_clusters=extracted_colour_count,
        random_state=42,
        n_init=30
    )

    labels = model.fit_predict(
        lab_colours,
        sample_weight=sample_weights
    )

    cluster_centres = (
        model.cluster_centers_
    )

    rows = []

    for cluster_index in range(
        extracted_colour_count
    ):
        member_indices = np.where(
            labels == cluster_index
        )[0]

        member_lab = lab_colours[
            member_indices
        ]

        centre_lab = cluster_centres[
            cluster_index
        ]

        distances = np.linalg.norm(
            member_lab - centre_lab,
            axis=1
        )

        closest_position = np.argmin(
            distances
        )

        representative_index = (
            member_indices[
                closest_position
            ]
        )

        representative_rgb = (
            rgb_colours[
                representative_index
            ]
        )

        representative_lab = (
            lab_colours[
                representative_index
            ]
        )

        cluster_weight = float(
            sample_weights[
                member_indices
            ].sum()
        )

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
            "_internal_weight":
                cluster_weight,
        })

    dataframe = pd.DataFrame(rows)

    # 按内部权重排序
    dataframe = dataframe.sort_values(
        by="_internal_weight",
        ascending=False
    ).reset_index(drop=True)

    # 删除内部权重，不写入最终 CSV
    dataframe = dataframe.drop(
        columns=["_internal_weight"]
    )

    # ========================================================
    # 加入 #105CF4 作为品牌方向候选色
    # ========================================================

    brand_rgb = np.array(
        [16, 92, 244],
        dtype=np.float32
    )

    brand_lab = rgb2lab(
        (
            brand_rgb / 255.0
        ).reshape(1, 1, 3)
    ).reshape(3)

    brand_candidate = pd.DataFrame([
        {
            "hex": "#105CF4",
            "red": 16,
            "green": 92,
            "blue": 244,
            "lab_l": round(
                float(brand_lab[0]),
                2
            ),
            "lab_a": round(
                float(brand_lab[1]),
                2
            ),
            "lab_b": round(
                float(brand_lab[2]),
                2
            ),
            "hue": round(
                calculate_hue(
                    brand_rgb
                ),
                2
            ),
            "saturation": round(
                calculate_saturation(
                    brand_rgb
                ),
                2
            ),
            "brightness": round(
                calculate_brightness(
                    brand_rgb
                ),
                2
            ),
        }
    ])

    # 合并图片提取色和品牌候选色
    dataframe = pd.concat(
        [
            dataframe,
            brand_candidate
        ],
        ignore_index=True
    )

    # 按明度排序，形成自然的深蓝到亮蓝色阶
    dataframe = dataframe.sort_values(
        by="brightness",
        ascending=True
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
# 8. 生成最终比例色板
# ============================================================

def create_palette_image(
    final_colours: pd.DataFrame
) -> None:
    """
    生成从 100 张海洋图片中提取的
    10 个主要蓝色候选色。

    每个色块等宽显示，不展示占比数字。
    """

    colour_count = len(final_colours)

    figure, axis = plt.subplots(
        figsize=(16, 5)
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
                f"{row['hex']}"
            ),
            horizontalalignment="center",
            verticalalignment="center",
            color=get_text_colour(rgb),
            fontsize=10,
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
            "Top 10 Ocean Colour Candidates\n"
            "Extracted from 100 reference images"
        ),
        fontsize=18,
        pad=20
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER
        / "top_10_ocean_colours.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"已生成候选色板：{output_path}"
    )


# ============================================================
# 9. 生成独立色卡
# ============================================================

def create_colour_cards(
    final_colours: pd.DataFrame
) -> None:
    """
    生成 10 个海洋蓝候选色的独立色卡。
    """

    colour_count = len(
        final_colours
    )

    figure, axes = plt.subplots(
        nrows=colour_count,
        ncols=1,
        figsize=(
            9,
            max(2, colour_count * 1.2)
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
                f"{row['red']}, "
                f"{row['green']}, "
                f"{row['blue']})"
            ),
            transform=axis.transAxes,
            verticalalignment="center",
            color=get_text_colour(rgb),
            fontsize=12,
            fontweight="bold"
        )

        axis.set_xticks([])
        axis.set_yticks([])

        for spine in axis.spines.values():
            spine.set_visible(False)

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER
        / "top_10_ocean_colour_cards.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"已生成候选色卡：{output_path}"
    )


# ============================================================
# 10. 主程序
# ============================================================

def main() -> None:
    """
    执行完整分析流程。
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

    # 第一次：分析每张图片
    (
        image_colours,
        all_colour_samples
    ) = analyse_all_images()

    image_colours_path = (
        OUTPUT_FOLDER
        / "colours_from_each_image.csv"
    )

    image_colours.to_csv(
        image_colours_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        "每张图片的颜色数据已保存："
        f"{image_colours_path}"
    )

    # 第二次：目标导向的最终聚类
    final_colours = (
        create_final_colour_clusters(
            all_colour_samples
        )
    )

    final_colours_path = (
        OUTPUT_FOLDER
        / "top_10_ocean_colour_candidates.csv"
    )

    final_colours.to_csv(
        final_colours_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "最终颜色结果已保存："
        f"{final_colours_path}"
    )

    

        # 生成图片
    create_palette_image(
        final_colours
    )

    create_colour_cards(
        final_colours
    )

    print()
    print("分析完成。")
    print()
    print(
        "以下为从 100 张海洋图片中提取的"
        "前 10 个蓝色候选色："
    )

    for _, row in final_colours.iterrows():
        print(
            f"{int(row['rank']):02d}. "
            f"{row['hex']} "
            f"RGB("
            f"{row['red']}, "
            f"{row['green']}, "
            f"{row['blue']})"
        )


if __name__ == "__main__":
    main()