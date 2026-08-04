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
MIN_SATURATION = 0.40

# 只保留蓝色色相
#
# 180° 左右：青色
# 190°～210°：天蓝、海蓝
# 210°～230°：标准蓝
# 230°～250°：深蓝、偏紫蓝
BLUE_HUE_MIN = 190
BLUE_HUE_MAX = 250
# ============================================================
# 品牌蓝校准参数
# ============================================================

TARGET_BLUE = "#105CF4"

# 防止平均后颜色变灰
SATURATION_BOOST = 1.18

# 防止平均后颜色变暗
VALUE_BOOST = 1.25

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
        np.asarray(
            rgb,
            dtype=np.float32
        )
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
        np.asarray(
            rgb,
            dtype=np.float32
        )
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
        np.asarray(
            rgb,
            dtype=np.float32
        )
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
        np.asarray(
            rgb,
            dtype=np.float32
        )
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


def calculate_weighted_circular_hue(
    hues: np.ndarray,
    weights: np.ndarray
) -> float:
    """
    计算加权平均色相。

    色相是一个圆形数值：
    0° 和 360° 实际上是同一个方向。

    虽然当前蓝色色相限制在 190°～250°，
    不会跨越 0°，但使用圆形平均算法更稳定。
    """

    hues = np.asarray(
        hues,
        dtype=np.float64
    )

    weights = np.asarray(
        weights,
        dtype=np.float64
    )

    if len(hues) == 0:
        raise ValueError(
            "没有可用于计算平均色相的数据。"
        )

    if len(hues) != len(weights):
        raise ValueError(
            "色相数量和权重数量不一致。"
        )

    valid_mask = (
        np.isfinite(hues)
        & np.isfinite(weights)
        & (weights > 0)
    )

    hues = hues[
        valid_mask
    ]

    weights = weights[
        valid_mask
    ]

    if len(hues) == 0:
        raise ValueError(
            "没有有效的色相和权重数据。"
        )

    weights = (
        weights / weights.sum()
    )

    radians = np.deg2rad(
        hues
    )

    weighted_sine = np.sum(
        weights * np.sin(radians)
    )

    weighted_cosine = np.sum(
        weights * np.cos(radians)
    )

    average_angle = np.arctan2(
        weighted_sine,
        weighted_cosine
    )

    average_hue = (
        np.rad2deg(average_angle)
        % 360
    )

    return float(
        average_hue
    )

def hsv_weighted_average_colour(
    final_colours: pd.DataFrame
) -> np.ndarray:
    """
    使用HSV空间计算最终品牌蓝。

    RGB/Lab平均容易降低饱和度。
    HSV平均可以保持高饱和蓝色。
    """

    weights = (
        final_colours[
            "frequency_percent"
        ]
        .to_numpy()
    )

    weights = (
        weights /
        weights.sum()
    )


    rgb_values = (
        final_colours[
            [
                "red",
                "green",
                "blue"
            ]
        ]
        .to_numpy()
    )


    hsv_values=[]


    for rgb in rgb_values:

        hsv_values.append(
            colorsys.rgb_to_hsv(
                rgb[0]/255,
                rgb[1]/255,
                rgb[2]/255
            )
        )


    hsv_values=np.array(
        hsv_values
    )


    # -----------------------
    # Hue圆形平均
    # -----------------------

    angles = (
        hsv_values[:,0]
        *
        2
        *
        np.pi
    )


    x=np.sum(
        np.cos(angles)
        *
        weights
    )


    y=np.sum(
        np.sin(angles)
        *
        weights
    )


    hue=(
        np.arctan2(y,x)
        /
        (2*np.pi)
    )


    if hue < 0:
        hue+=1



    # -----------------------
    # S 和 V 加权平均
    # -----------------------

    saturation=np.average(
        hsv_values[:,1],
        weights=weights
    )


    value=np.average(
        hsv_values[:,2],
        weights=weights
    )



    # -----------------------
    # 品牌蓝增强
    # -----------------------

    saturation*=SATURATION_BOOST

    value*=VALUE_BOOST


    saturation=np.clip(
        saturation,
        0,
        1
    )


    value=np.clip(
        value,
        0,
        1
    )


    rgb=colorsys.hsv_to_rgb(
        hue,
        saturation,
        value
    )


    return (
        np.array(rgb)
        *
        255
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

        filtered_pixels = filtered_pixels[
            selected_indices
        ]

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
                    4
                ),
                "lab_a": round(
                    float(lab[1]),
                    4
                ),
                "lab_b": round(
                    float(lab[2]),
                    4
                ),
            })

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

    image_colour_dataframe = pd.DataFrame(
        csv_rows
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
    从所有图片的颜色样本中聚类出最终 10 个蓝色。

    最终结果：

    1. 所有颜色均来自图片；
    2. 使用原图颜色占比作为权重；
    3. 权重重新归一化，总和为 100%；
    4. 最终色卡按照色相排列；
    5. 同时保留颜色的权重排名。
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

    blue_mask = (
        (hue >= BLUE_HUE_MIN)
        & (hue <= BLUE_HUE_MAX)
        & (saturation >= MIN_SATURATION)
        & (value >= MIN_VALUE)
        & (value <= MAX_VALUE)
        & np.isfinite(sample_weights)
        & (sample_weights > 0)
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

    unique_lab_colours = np.unique(
        np.round(
            lab_colours,
            decimals=3
        ),
        axis=0
    )

    actual_cluster_count = min(
        FINAL_COLOUR_COUNT,
        len(lab_colours),
        len(unique_lab_colours)
    )

    if actual_cluster_count < 1:
        raise ValueError(
            "没有足够的蓝色样本进行最终聚类。"
        )

    model = KMeans(
        n_clusters=actual_cluster_count,
        random_state=42,
        n_init=30
    )

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

    if total_weight <= 0:
        raise ValueError(
            "蓝色样本权重总和小于或等于零。"
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

        distances = np.linalg.norm(
            member_lab - centre_lab,
            axis=1
        )

        closest_position = int(
            np.argmin(distances)
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
            * 100.0
        )

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
            "hue": calculate_hue(
                representative_rgb
            ),
            "saturation": calculate_saturation(
                representative_rgb
            ),
            "brightness": calculate_brightness(
                representative_rgb
            ),
            "raw_weight": cluster_weight,
            "frequency_percent": cluster_percentage,
            "sample_count": int(
                len(member_indices)
            ),
            "lab_l": float(
                representative_lab[0]
            ),
            "lab_a": float(
                representative_lab[1]
            ),
            "lab_b": float(
                representative_lab[2]
            ),
        })

    dataframe = pd.DataFrame(
        rows
    )

    if dataframe.empty:
        raise ValueError(
            "最终没有得到符合条件的蓝色。"
        )

    visible_weight_sum = dataframe[
        "frequency_percent"
    ].sum()

    if visible_weight_sum <= 0:
        raise ValueError(
            "最终颜色权重总和异常。"
        )

    dataframe[
        "frequency_percent"
    ] = (
        dataframe[
            "frequency_percent"
        ]
        / visible_weight_sum
        * 100.0
    )

    dataframe[
        "weight_rank"
    ] = (
        dataframe[
            "frequency_percent"
        ]
        .rank(
            method="first",
            ascending=False
        )
        .astype(int)
    )

    dataframe = dataframe.sort_values(
        by=[
            "hue",
            "brightness",
            "saturation"
        ],
        ascending=[
            True,
            False,
            False
        ]
    ).reset_index(
        drop=True
    )

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
# 8. 计算最终加权平均蓝色
# ============================================================

def create_weighted_average_colour(
    final_colours: pd.DataFrame
) -> dict:
    """
    根据最终蓝色的权重，计算一个最终平均蓝色。

    同时计算：

    1. 加权平均色相；
    2. 加权平均饱和度；
    3. 加权平均明度；
    4. Lab 色彩空间中的加权平均代表色。

    最终 HEX 使用 Lab 加权平均色计算。
    """

    if final_colours.empty:
        raise ValueError(
            "没有最终颜色，无法计算平均颜色。"
        )

    weights = final_colours[
        "frequency_percent"
    ].to_numpy(
        dtype=np.float64
    )

    if (
        not np.all(np.isfinite(weights))
        or weights.sum() <= 0
    ):
        raise ValueError(
            "最终颜色权重无效。"
        )

    weights = (
        weights / weights.sum()
    )

    hues = final_colours[
        "hue"
    ].to_numpy(
        dtype=np.float64
    )

    saturations = final_colours[
        "saturation"
    ].to_numpy(
        dtype=np.float64
    )

    brightness_values = final_colours[
        "brightness"
    ].to_numpy(
        dtype=np.float64
    )

    lab_values = final_colours[
        [
            "lab_l",
            "lab_a",
            "lab_b"
        ]
    ].to_numpy(
        dtype=np.float64
    )

    # 色相需要使用圆形平均
    weighted_average_hue = (
        calculate_weighted_circular_hue(
            hues,
            weights
        )
    )

    weighted_average_saturation = float(
        np.average(
            saturations,
            weights=weights
        )
    )

    weighted_average_brightness = float(
        np.average(
            brightness_values,
            weights=weights
        )
    )


    # ==================================================
    # 使用 HSV 平均生成最终品牌蓝
    # ==================================================

    weighted_average_rgb = (
        hsv_weighted_average_colour(
            final_colours
        )
    )


    weighted_average_rgb = np.clip(
        weighted_average_rgb,
        0,
        255
    )


    # 根据最终RGB重新计算Lab
    # 只用于保存数据，不参与颜色生成

    weighted_average_lab = rgb2lab(
        (
            weighted_average_rgb
            /
            255.0
        )
        .reshape(
            1,
            1,
            3
        )
    ).reshape(3)

    # Lab 平均颜色转换成 RGB 后，
    # 再计算它自身实际对应的 HSV 数值
    actual_rgb_hue = calculate_hue(
        weighted_average_rgb
    )

    actual_rgb_saturation = (
        calculate_saturation(
            weighted_average_rgb
        )
    )

    actual_rgb_brightness = (
        calculate_brightness(
            weighted_average_rgb
        )
    )

    return {
        "hex": rgb_to_hex(
            weighted_average_rgb
        ),
        "rgb": weighted_average_rgb,
        "red": int(
            round(
                float(
                    weighted_average_rgb[0]
                )
            )
        ),
        "green": int(
            round(
                float(
                    weighted_average_rgb[1]
                )
            )
        ),
        "blue": int(
            round(
                float(
                    weighted_average_rgb[2]
                )
            )
        ),
        "weighted_average_hue": (
            weighted_average_hue
        ),
        "weighted_average_saturation": (
            weighted_average_saturation
        ),
        "weighted_average_brightness": (
            weighted_average_brightness
        ),
        "actual_rgb_hue": (
            actual_rgb_hue
        ),
        "actual_rgb_saturation": (
            actual_rgb_saturation
        ),
        "actual_rgb_brightness": (
            actual_rgb_brightness
        ),
        "lab_l": float(
            weighted_average_lab[0]
        ),
        "lab_a": float(
            weighted_average_lab[1]
        ),
        "lab_b": float(
            weighted_average_lab[2]
        ),
    }


# ============================================================
# 9. 生成横向色板
# ============================================================

def create_palette_image(
    final_colours: pd.DataFrame
) -> None:
    """
    生成横向蓝色色板。

    色卡按照色相排列：
    青蓝 -> 标准蓝 -> 偏紫蓝。
    """

    hue_sorted_colours = (
        final_colours.sort_values(
            by=[
                "hue",
                "brightness",
                "saturation"
            ],
            ascending=[
                True,
                False,
                False
            ]
        ).reset_index(drop=True)
    )

    colour_count = len(
        hue_sorted_colours
    )

    figure, axis = plt.subplots(
        figsize=(18, 5)
    )

    for position, (_, row) in enumerate(
        hue_sorted_colours.iterrows()
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
                f"{position + 1:02d}\n"
                f"{row['hex']}\n"
                f"H {row['hue']:.0f}°\n"
                f"{row['frequency_percent']:.1f}%"
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
            "Sorted by Hue · Weight Shown on Each Colour"
        ),
        fontsize=18,
        pad=20
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER
        / "top_10_blue_colours_hue_sorted.png"
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
        f"已生成按色相排列的蓝色色板：{output_path}"
    )


# ============================================================
# 10. 生成独立色卡
# ============================================================

def create_colour_cards(
    final_colours: pd.DataFrame
) -> None:
    """
    生成独立蓝色色卡。

    顺序按照色相排列。
    每张色卡显示颜色权重和权重排名。
    """

    hue_sorted_colours = (
        final_colours.sort_values(
            by=[
                "hue",
                "brightness",
                "saturation"
            ],
            ascending=[
                True,
                False,
                False
            ]
        ).reset_index(drop=True)
    )

    colour_count = len(
        hue_sorted_colours
    )

    figure, axes = plt.subplots(
        nrows=colour_count,
        ncols=1,
        figsize=(
            12,
            max(
                3,
                colour_count * 1.25
            )
        )
    )

    if colour_count == 1:
        axes = [axes]

    for position, (
        axis,
        (_, row)
    ) in enumerate(
        zip(
            axes,
            hue_sorted_colours.iterrows()
        ),
        start=1
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
                f"{position:02d}   "
                f"{row['hex']}   "
                f"RGB("
                f"{int(row['red'])}, "
                f"{int(row['green'])}, "
                f"{int(row['blue'])})   "
                f"H {row['hue']:.1f}°   "
                f"S {row['saturation']:.1f}%   "
                f"V {row['brightness']:.1f}%   "
                f"Weight {row['frequency_percent']:.2f}%   "
                f"Weight Rank #{int(row['weight_rank'])}"
            ),
            transform=axis.transAxes,
            verticalalignment="center",
            color=get_text_colour(rgb),
            fontsize=10.5,
            fontweight="bold"
        )

        axis.set_xticks([])
        axis.set_yticks([])

        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.suptitle(
        (
            "Blue Colour Cards Sorted by Hue\n"
            "Cyan Blue → Standard Blue → Violet Blue"
        ),
        fontsize=16,
        y=1.01
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER
        / "top_10_blue_colour_cards_hue_sorted.png"
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
        f"已生成按色相排列的蓝色色卡：{output_path}"
    )


# ============================================================
# 11. 生成按出现频率排列的色板
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

    frequencies = frequency_colours[
        "frequency_percent"
    ].to_numpy(
        dtype=np.float64
    )

    frequency_sum = frequencies.sum()

    if frequency_sum <= 0:
        frequencies = np.ones(
            len(frequency_colours)
        )

        frequency_sum = frequencies.sum()

    figure, axis = plt.subplots(
        figsize=(18, 5)
    )

    current_left = 0.0

    for _, row in frequency_colours.iterrows():
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
# 12. 生成最终平均蓝色色卡
# ============================================================

def create_average_colour_card(
    average_colour: dict
) -> None:
    """
    为最终 Lab 加权平均蓝色生成一张独立色卡。
    """

    rgb = np.asarray(
        average_colour["rgb"],
        dtype=np.float32
    )

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.set_facecolor(
        rgb / 255.0
    )

    axis.text(
        0.5,
        0.58,
        average_colour["hex"],
        transform=axis.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        color=get_text_colour(rgb),
        fontsize=38,
        fontweight="bold"
    )

    axis.text(
        0.5,
        0.37,
        (
            f"RGB("
            f"{average_colour['red']}, "
            f"{average_colour['green']}, "
            f"{average_colour['blue']})\n"
            f"Weighted Average Hue "
            f"{average_colour['weighted_average_hue']:.2f}°\n"
            f"Lab Average Colour Hue "
            f"{average_colour['actual_rgb_hue']:.2f}°"
        ),
        transform=axis.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        color=get_text_colour(rgb),
        fontsize=15,
        fontweight="bold",
        linespacing=1.5
    )

    axis.set_xticks([])
    axis.set_yticks([])

    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.set_title(
        "Final Weighted Average Blue",
        fontsize=20,
        pad=20
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER
        / "weighted_average_blue.png"
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
        f"已生成最终平均蓝色色卡：{output_path}"
    )


# ============================================================
# 13. 保存最终平均颜色数据
# ============================================================

def save_average_colour_data(
    average_colour: dict
) -> None:
    """
    保存最终平均蓝色的数据。
    """

    average_colour_dataframe = pd.DataFrame([
        {
            "hex": average_colour["hex"],
            "red": average_colour["red"],
            "green": average_colour["green"],
            "blue": average_colour["blue"],
            "weighted_average_hue": round(
                average_colour[
                    "weighted_average_hue"
                ],
                4
            ),
            "weighted_average_saturation": round(
                average_colour[
                    "weighted_average_saturation"
                ],
                4
            ),
            "weighted_average_brightness": round(
                average_colour[
                    "weighted_average_brightness"
                ],
                4
            ),
            "lab_average_colour_hue": round(
                average_colour[
                    "actual_rgb_hue"
                ],
                4
            ),
            "lab_average_colour_saturation": round(
                average_colour[
                    "actual_rgb_saturation"
                ],
                4
            ),
            "lab_average_colour_brightness": round(
                average_colour[
                    "actual_rgb_brightness"
                ],
                4
            ),
            "lab_l": round(
                average_colour["lab_l"],
                4
            ),
            "lab_a": round(
                average_colour["lab_a"],
                4
            ),
            "lab_b": round(
                average_colour["lab_b"],
                4
            ),
        }
    ])

    output_path = (
        OUTPUT_FOLDER
        / "weighted_average_blue.csv"
    )

    average_colour_dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"最终平均蓝色数据已保存：{output_path}"
    )


# ============================================================
# 14. 主程序
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

    # 输出 CSV 时单独四舍五入，
    # 不修改程序中用于计算平均值的原始数据。
    final_colours_for_csv = (
        final_colours.copy()
    )

    decimal_columns = [
        "hue",
        "saturation",
        "brightness",
        "raw_weight",
        "frequency_percent",
        "lab_l",
        "lab_a",
        "lab_b",
    ]

    final_colours_for_csv[
        decimal_columns
    ] = final_colours_for_csv[
        decimal_columns
    ].round(4)

    final_colours_for_csv.to_csv(
        final_colours_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "最终蓝色结果已保存："
        f"{final_colours_path}"
    )

    # --------------------------------------------------------
    # 第三步：计算最终平均色相和平均蓝色
    # --------------------------------------------------------

    average_colour = (
        create_weighted_average_colour(
            final_colours
        )
    )

    save_average_colour_data(
        average_colour
    )

    # --------------------------------------------------------
    # 第四步：生成可视化图片
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

    create_average_colour_card(
        average_colour
    )

    # --------------------------------------------------------
    # 第五步：终端打印全部结果
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("分析完成")
    print("=" * 90)

    hue_sorted_colours = (
        final_colours.sort_values(
            by=[
                "hue",
                "brightness",
                "saturation"
            ],
            ascending=[
                True,
                False,
                False
            ]
        ).reset_index(drop=True)
    )

    print()
    print(
        "按色相排列："
        "青蓝 -> 海蓝 -> 标准蓝 -> 偏紫蓝"
    )

    print()

    for position, (_, row) in enumerate(
        hue_sorted_colours.iterrows(),
        start=1
    ):
        print(
            f"{position:02d}. "
            f"{row['hex']}  "
            f"RGB("
            f"{int(row['red'])}, "
            f"{int(row['green'])}, "
            f"{int(row['blue'])})  "
            f"H={row['hue']:.1f}°  "
            f"S={row['saturation']:.1f}%  "
            f"V={row['brightness']:.1f}%  "
            f"Weight={row['frequency_percent']:.2f}%  "
            f"Weight Rank=#{int(row['weight_rank'])}"
        )

    print()
    print("-" * 90)
    print("按权重从高到低排列")
    print("-" * 90)
    print()

    weight_sorted_colours = (
        final_colours.sort_values(
            by="frequency_percent",
            ascending=False
        ).reset_index(drop=True)
    )

    for position, (_, row) in enumerate(
        weight_sorted_colours.iterrows(),
        start=1
    ):
        print(
            f"{position:02d}. "
            f"{row['hex']}  "
            f"Weight={row['frequency_percent']:.2f}%  "
            f"H={row['hue']:.1f}°  "
            f"Samples={int(row['sample_count'])}"
        )

    total_percentage = float(
        final_colours[
            "frequency_percent"
        ].sum()
    )

    print()
    print(
        f"全部颜色权重总和："
        f"{total_percentage:.2f}%"
    )

    print()
    print("=" * 90)
    print("最终加权平均结果")
    print("=" * 90)
    print()

    print(
        "10 个颜色的加权平均色相："
        f"{average_colour['weighted_average_hue']:.2f}°"
    )

    print(
        "10 个颜色的加权平均饱和度："
        f"{average_colour['weighted_average_saturation']:.2f}%"
    )

    print(
        "10 个颜色的加权平均明度："
        f"{average_colour['weighted_average_brightness']:.2f}%"
    )

    print()
    print(
        " HSV加权平均生成的最终代表蓝色："
    )

    print(
        f"HEX：{average_colour['hex']}"
    )

    print(
        "RGB："
        f"RGB("
        f"{average_colour['red']}, "
        f"{average_colour['green']}, "
        f"{average_colour['blue']})"
    )

    print(
        "最终代表色实际 HSV："
        f"H={average_colour['actual_rgb_hue']:.2f}°  "
        f"S={average_colour['actual_rgb_saturation']:.2f}%  "
        f"V={average_colour['actual_rgb_brightness']:.2f}%"
    )

    print(
        "最终代表色 Lab："
        f"L={average_colour['lab_l']:.2f}  "
        f"a={average_colour['lab_a']:.2f}  "
        f"b={average_colour['lab_b']:.2f}"
    )

    print()
    print(
        "结果文件夹："
        f"{OUTPUT_FOLDER.resolve()}"
    )


if __name__ == "__main__":
    main()