import csv
import io
import os
import time
import colorsys
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError


# ============================================================
# 1. 基础设置
# ============================================================

# 不要把 Access Key 直接写进代码或上传到 GitHub。
#
# Mac / Linux 终端运行：
# export UNSPLASH_ACCESS_KEY="你的 Unsplash Access Key"
#
# Windows PowerShell 运行：
# $env:UNSPLASH_ACCESS_KEY="你的 Unsplash Access Key"

ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

if not ACCESS_KEY:
    raise ValueError(
        "没有找到 UNSPLASH_ACCESS_KEY。\n"
        "请先在终端设置 Unsplash Access Key。"
    )


# 目标品牌色
TARGET_COLOR = "#105CF4"

# 最终图片保存位置
SAVE_FOLDER = Path("images")

# 图片评分结果
RESULT_CSV = Path("image_colour_scores.csv")

# 最终需要保存的图片数量
TARGET_COUNT = 100

# 最多分析多少张候选图片
# 如果最后筛出的图片太少，可以改成 800 或 1000
MAX_CANDIDATES = 500

# 每个关键词最多搜索多少页
MAX_PAGES_PER_KEYWORD = 5

# Unsplash 每页最多返回 30 张
PER_PAGE = 30

# 用于颜色分析的小图宽度
PREVIEW_WIDTH = 400

# 最终保存图片的宽度
FINAL_IMAGE_WIDTH = 1600

# 请求间隔，避免连续请求过快
REQUEST_DELAY = 0.12

# 自动创建图片文件夹
SAVE_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. 搜索关键词
# ============================================================

# 尽量搜索大面积、高饱和蓝色水体。
# 避免 beach、sunset、coast、sky 等容易带来其他颜色的词。

KEYWORDS = [
    "vivid cobalt blue underwater",
    "deep royal blue underwater",
    "saturated blue ocean underwater",
    "electric blue underwater ocean",
    "intense cobalt blue sea",
    "ultramarine ocean water texture",
    "deep blue ocean water close up",
    "royal blue sea surface texture",
    "vivid blue underwater light",
    "dark saturated blue ocean",
    "cobalt blue water abstract",
    "high saturation blue sea",
    "deep blue underwater photography",
    "blue ocean texture no sky",
    "cobalt blue underwater animal",
    "deep ocean blue glow",
    "ultramarine water close up",
    "intense blue underwater photography",
]


# ============================================================
# 3. 图片筛选参数
# ============================================================

# 你喜欢的图片主要集中在：
# 高饱和钴蓝、皇家蓝、电光蓝和深蓝。

# HSV 蓝色色相范围
MIN_BLUE_HUE = 0.54
MAX_BLUE_HUE = 0.70

# 一个像素被认为是蓝色时，最低饱和度
MIN_BLUE_PIXEL_SATURATION = 0.45

# 高饱和像素的判定标准
HIGH_SATURATION_THRESHOLD = 0.62

# 图片中蓝色像素最低占比
MIN_BLUE_RATIO = 0.60

# 图片中高饱和像素最低占比
MIN_HIGH_SATURATION_RATIO = 0.52

# 更贴近 #105CF4 的核心蓝色色相范围
CORE_BLUE_HUE_MIN = 0.58
CORE_BLUE_HUE_MAX = 0.67

# 图片中核心蓝色最低占比
MIN_CORE_BLUE_RATIO = 0.20

# 图片整体最低平均饱和度
MIN_MEAN_SATURATION = 0.55

# 蓝色区域内部最低平均饱和度
MIN_BLUE_SATURATION = 0.62

# 允许有高饱和深蓝，但限制无色黑色区域
MAX_NEUTRAL_DARK_RATIO = 0.18

# 允许少量白色反光或浪花
MAX_WHITE_RATIO = 0.12

# 允许少量灰色和低饱和区域
MAX_GREY_RATIO = 0.18

# 偏青绿色区域最大比例
MAX_CYAN_GREEN_RATIO = 0.20

# 偏紫区域最大比例
MAX_PURPLE_RATIO = 0.12


# ============================================================
# 4. 基础颜色函数
# ============================================================

def hex_to_rgb(hex_colour: str) -> tuple[int, int, int]:
    """把十六进制颜色转换为 RGB。"""

    clean_hex = hex_colour.lstrip("#")

    if len(clean_hex) != 6:
        raise ValueError(
            f"无效的十六进制颜色：{hex_colour}"
        )

    return (
        int(clean_hex[0:2], 16),
        int(clean_hex[2:4], 16),
        int(clean_hex[4:6], 16),
    )


def get_target_hsv(
    hex_colour: str,
) -> tuple[float, float, float]:
    """获得目标颜色的 HSV 数值，范围为 0～1。"""

    red, green, blue = hex_to_rgb(hex_colour)

    return colorsys.rgb_to_hsv(
        red / 255,
        green / 255,
        blue / 255,
    )


TARGET_HUE, TARGET_SATURATION, TARGET_VALUE = get_target_hsv(
    TARGET_COLOR
)


def circular_hue_distance(
    hue_value: float,
    target_hue: float,
) -> float:
    """
    计算两个 HSV 色相之间的环形距离。

    HSV 色相的 0 和 1 是相邻的。
    """

    difference = abs(
        hue_value - target_hue
    )

    return min(
        difference,
        1.0 - difference,
    )


# ============================================================
# 5. 网络请求和图片读取
# ============================================================

def request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any] | None:
    """发送请求并返回 JSON。"""

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:
        print(f"网络请求失败：{error}")
        return None

    except ValueError as error:
        print(f"JSON 解析失败：{error}")
        return None


def request_image_bytes(
    url: str,
    timeout: int = 30,
) -> bytes | None:
    """请求图片并返回二进制内容。"""

    try:
        response = requests.get(
            url,
            timeout=timeout,
        )

        response.raise_for_status()

        return response.content

    except requests.RequestException as error:
        print(f"\n图片请求失败：{error}")
        return None


def open_image_from_bytes(
    image_bytes: bytes,
) -> Image.Image | None:
    """从二进制内容打开图片。"""

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        return image

    except (
        UnidentifiedImageError,
        OSError,
    ) as error:
        print(f"\n图片无法读取：{error}")
        return None


def build_unsplash_image_url(
    raw_url: str,
    *,
    width: int,
    quality: int,
) -> str:
    """构造指定宽度和质量的 Unsplash 图片地址。"""

    separator = "&" if "?" in raw_url else "?"

    return (
        f"{raw_url}"
        f"{separator}w={width}"
        f"&q={quality}"
        f"&fit=max"
        f"&auto=format"
    )


# ============================================================
# 6. 分析图片颜色
# ============================================================

def analyse_image_colour(
    image: Image.Image,
) -> dict[str, float | bool]:
    """
    筛选高饱和蓝色海洋图片。

    重点保留：
    - 钴蓝
    - 皇家蓝
    - 电光蓝
    - 高饱和深蓝

    排除：
    - 灰蓝色
    - 大面积白色浪花
    - 无色黑色
    - 青绿色海水
    - 大面积紫色
    """

    preview = image.copy()

    preview.thumbnail(
        (PREVIEW_WIDTH, PREVIEW_WIDTH)
    )

    hsv_image = preview.convert("HSV")

    hsv_array = np.asarray(
        hsv_image,
        dtype=np.float32,
    ) / 255.0

    hue = hsv_array[:, :, 0]
    saturation = hsv_array[:, :, 1]
    value = hsv_array[:, :, 2]

    # --------------------------------------------------------
    # 蓝色区域
    # --------------------------------------------------------

    blue_mask = (
        (hue >= MIN_BLUE_HUE)
        & (hue <= MAX_BLUE_HUE)
        & (saturation >= MIN_BLUE_PIXEL_SATURATION)
        & (value >= 0.04)
    )

    # 更贴近目标品牌蓝色的核心区域
    core_blue_mask = (
        (hue >= CORE_BLUE_HUE_MIN)
        & (hue <= CORE_BLUE_HUE_MAX)
        & (saturation >= 0.55)
        & (value >= 0.05)
    )

    # 高饱和像素
    high_saturation_mask = (
        saturation >= HIGH_SATURATION_THRESHOLD
    )

    # 高饱和蓝色
    vivid_blue_mask = (
        blue_mask
        & high_saturation_mask
    )

    # --------------------------------------------------------
    # 不希望出现的区域
    # --------------------------------------------------------

    # 无色黑色
    # 高饱和深蓝不会被算作坏暗部
    neutral_dark_mask = (
        (value < 0.12)
        & (saturation < 0.35)
    )

    # 接近纯白的高光或浪花
    white_mask = (
        (value > 0.92)
        & (saturation < 0.20)
    )

    # 灰色和低饱和区域
    grey_mask = (
        saturation < 0.22
    )

    # 偏青绿色
    cyan_green_mask = (
        (hue > 0.42)
        & (hue < MIN_BLUE_HUE)
        & (saturation > 0.35)
    )

    # 偏紫色
    purple_mask = (
        (hue > MAX_BLUE_HUE)
        & (hue < 0.82)
        & (saturation > 0.35)
    )

    # --------------------------------------------------------
    # 计算区域比例
    # --------------------------------------------------------

    blue_ratio = float(
        np.mean(blue_mask)
    )

    core_blue_ratio = float(
        np.mean(core_blue_mask)
    )

    vivid_blue_ratio = float(
        np.mean(vivid_blue_mask)
    )

    high_saturation_ratio = float(
        np.mean(high_saturation_mask)
    )

    neutral_dark_ratio = float(
        np.mean(neutral_dark_mask)
    )

    white_ratio = float(
        np.mean(white_mask)
    )

    grey_ratio = float(
        np.mean(grey_mask)
    )

    cyan_green_ratio = float(
        np.mean(cyan_green_mask)
    )

    purple_ratio = float(
        np.mean(purple_mask)
    )

    mean_saturation = float(
        np.mean(saturation)
    )

    mean_value = float(
        np.mean(value)
    )

    # --------------------------------------------------------
    # 蓝色区域内部数据
    # --------------------------------------------------------

    if np.any(blue_mask):
        blue_saturation = float(
            np.mean(
                saturation[blue_mask]
            )
        )

        blue_value = float(
            np.mean(
                value[blue_mask]
            )
        )

        weights = (
            saturation[blue_mask]
            * (0.20 + value[blue_mask])
        )

        if float(np.sum(weights)) > 0:
            mean_blue_hue = float(
                np.average(
                    hue[blue_mask],
                    weights=weights,
                )
            )
        else:
            mean_blue_hue = float(
                np.mean(
                    hue[blue_mask]
                )
            )

    else:
        blue_saturation = 0.0
        blue_value = 0.0
        mean_blue_hue = 0.0

    # --------------------------------------------------------
    # 与目标色相的接近程度
    # --------------------------------------------------------

    target_hue_distance = circular_hue_distance(
        mean_blue_hue,
        TARGET_HUE,
    )

    target_hue_similarity = max(
        0.0,
        1.0 - target_hue_distance / 0.12,
    )

    # --------------------------------------------------------
    # 评分
    # --------------------------------------------------------

    score = (
        # 蓝色面积
        blue_ratio * 35

        # 高饱和蓝色面积
        + vivid_blue_ratio * 30

        # 核心皇家蓝、钴蓝面积
        + core_blue_ratio * 18

        # 整体饱和度
        + mean_saturation * 12

        # 蓝色区域饱和度
        + blue_saturation * 10

        # 与目标色相的接近程度
        + target_hue_similarity * 10

        # 无色暗部扣分
        - neutral_dark_ratio * 22

        # 白色区域扣分
        - white_ratio * 20

        # 灰色区域扣分
        - grey_ratio * 20

        # 青绿色区域扣分
        - cyan_green_ratio * 16

        # 紫色区域扣分
        - purple_ratio * 12
    )

    # --------------------------------------------------------
    # 是否通过筛选
    # --------------------------------------------------------

    passed = (
        blue_ratio >= MIN_BLUE_RATIO
        and high_saturation_ratio >= MIN_HIGH_SATURATION_RATIO
        and core_blue_ratio >= MIN_CORE_BLUE_RATIO
        and mean_saturation >= MIN_MEAN_SATURATION
        and blue_saturation >= MIN_BLUE_SATURATION
        and neutral_dark_ratio <= MAX_NEUTRAL_DARK_RATIO
        and white_ratio <= MAX_WHITE_RATIO
        and grey_ratio <= MAX_GREY_RATIO
        and cyan_green_ratio <= MAX_CYAN_GREEN_RATIO
        and purple_ratio <= MAX_PURPLE_RATIO
    )

    return {
        "passed": passed,
        "score": score,

        "blue_ratio": blue_ratio,
        "core_blue_ratio": core_blue_ratio,
        "vivid_blue_ratio": vivid_blue_ratio,
        "high_saturation_ratio": high_saturation_ratio,

        "mean_saturation": mean_saturation,
        "mean_value": mean_value,

        "blue_saturation": blue_saturation,
        "blue_value": blue_value,
        "mean_blue_hue": mean_blue_hue,

        "target_hue_distance": target_hue_distance,
        "target_hue_similarity": target_hue_similarity,

        "neutral_dark_ratio": neutral_dark_ratio,
        "white_ratio": white_ratio,
        "grey_ratio": grey_ratio,
        "cyan_green_ratio": cyan_green_ratio,
        "purple_ratio": purple_ratio,
    }


# ============================================================
# 7. 搜索 Unsplash 候选图片
# ============================================================

def search_candidates() -> list[dict[str, Any]]:
    """从 Unsplash 搜索候选图片。"""

    headers = {
        "Authorization": f"Client-ID {ACCESS_KEY}"
    }

    search_url = "https://api.unsplash.com/search/photos"

    candidates: list[dict[str, Any]] = []

    collected_photo_ids: set[str] = set()

    for keyword in KEYWORDS:
        if len(candidates) >= MAX_CANDIDATES:
            break

        print(f"\n正在搜索：{keyword}")

        for page in range(
            1,
            MAX_PAGES_PER_KEYWORD + 1,
        ):
            if len(candidates) >= MAX_CANDIDATES:
                break

            params = {
                "query": keyword,
                "page": page,
                "per_page": PER_PAGE,
                "orientation": "landscape",
                "content_filter": "high",
                "color": "blue",
            }

            data = request_json(
                search_url,
                params=params,
                headers=headers,
                timeout=30,
            )

            if data is None:
                print(
                    f"关键词搜索失败，跳过：{keyword}"
                )
                break

            photos = data.get(
                "results",
                [],
            )

            if not photos:
                break

            for photo in photos:
                if len(candidates) >= MAX_CANDIDATES:
                    break

                photo_id = photo.get("id")

                if not photo_id:
                    continue

                if photo_id in collected_photo_ids:
                    continue

                urls = photo.get(
                    "urls",
                    {},
                )

                raw_url = urls.get("raw")

                if not raw_url:
                    continue

                collected_photo_ids.add(photo_id)

                preview_url = build_unsplash_image_url(
                    raw_url,
                    width=PREVIEW_WIDTH,
                    quality=60,
                )

                photographer = (
                    photo.get("user", {})
                    .get("name", "")
                )

                download_location = (
                    photo.get("links", {})
                    .get("download_location")
                )

                candidates.append({
                    "id": photo_id,
                    "keyword": keyword,
                    "preview_url": preview_url,
                    "raw_url": raw_url,
                    "download_location": download_location,
                    "photographer": photographer,
                })

            print(
                f"已收集候选图片："
                f"{len(candidates)}/{MAX_CANDIDATES}"
            )

            time.sleep(REQUEST_DELAY)

    return candidates


# ============================================================
# 8. 给候选图片评分
# ============================================================

def score_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """分析所有候选图片的实际颜色。"""

    accepted: list[dict[str, Any]] = []

    total = len(candidates)

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"\r正在分析颜色：{index}/{total}",
            end="",
            flush=True,
        )

        image_bytes = request_image_bytes(
            candidate["preview_url"]
        )

        if image_bytes is None:
            continue

        image = open_image_from_bytes(
            image_bytes
        )

        if image is None:
            continue

        colour_result = analyse_image_colour(
            image
        )

        candidate.update(
            colour_result
        )

        if bool(colour_result["passed"]):
            accepted.append(candidate)

        time.sleep(REQUEST_DELAY)

    print()

    accepted.sort(
        key=lambda item: float(
            item.get("score", 0)
        ),
        reverse=True,
    )

    print(
        f"\n通过颜色筛选的图片："
        f"{len(accepted)}/{len(candidates)}"
    )

    return accepted


# ============================================================
# 9. Unsplash 下载统计
# ============================================================

def trigger_unsplash_download(
    download_location: str | None,
) -> None:
    """
    根据 Unsplash API 规范触发下载统计。

    这不会保存图片，只用于通知 Unsplash
    某张图片被下载。
    """

    if not download_location:
        return

    headers = {
        "Authorization": f"Client-ID {ACCESS_KEY}"
    }

    try:
        requests.get(
            download_location,
            headers=headers,
            timeout=15,
        )

    except requests.RequestException:
        pass


# ============================================================
# 10. 保存最终图片
# ============================================================

def download_final_images(
    accepted: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按照评分从高到低下载最终图片。"""

    selected = accepted[:TARGET_COUNT]

    saved_results: list[dict[str, Any]] = []

    for index, candidate in enumerate(
        selected,
        start=1,
    ):
        filename = SAVE_FOLDER / (
            f"sea_{index:03d}.jpg"
        )

        raw_url = str(
            candidate["raw_url"]
        )

        final_url = build_unsplash_image_url(
            raw_url,
            width=FINAL_IMAGE_WIDTH,
            quality=85,
        )

        print(
            f"正在下载 {filename.name}"
            f" | 评分：{float(candidate['score']):.2f}"
            f" | 蓝色占比："
            f"{float(candidate['blue_ratio']):.1%}"
            f" | 高饱和蓝："
            f"{float(candidate['vivid_blue_ratio']):.1%}"
        )

        image_bytes = request_image_bytes(
            final_url
        )

        if image_bytes is None:
            continue

        image = open_image_from_bytes(
            image_bytes
        )

        if image is None:
            continue

        try:
            # 统一转成 RGB JPEG
            image.save(
                filename,
                format="JPEG",
                quality=90,
                optimize=True,
            )

        except OSError as error:
            print(
                f"文件保存失败：{error}"
            )
            continue

        trigger_unsplash_download(
            candidate.get(
                "download_location"
            )
        )

        candidate["filename"] = filename.name

        saved_results.append(
            candidate
        )

        time.sleep(REQUEST_DELAY)

    return saved_results


# ============================================================
# 11. 输出评分 CSV
# ============================================================

def save_score_csv(
    results: list[dict[str, Any]],
) -> None:
    """保存每张最终图片的颜色评分。"""

    fieldnames = [
        "filename",
        "id",
        "keyword",
        "photographer",
        "score",

        "blue_ratio",
        "core_blue_ratio",
        "vivid_blue_ratio",
        "high_saturation_ratio",

        "mean_saturation",
        "mean_value",

        "blue_saturation",
        "blue_value",
        "mean_blue_hue",

        "target_hue_distance",
        "target_hue_similarity",

        "neutral_dark_ratio",
        "white_ratio",
        "grey_ratio",
        "cyan_green_ratio",
        "purple_ratio",
    ]

    try:
        with open(
            RESULT_CSV,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for result in results:
                row = {
                    field: result.get(
                        field,
                        "",
                    )
                    for field in fieldnames
                }

                writer.writerow(row)

    except OSError as error:
        print(
            f"CSV 保存失败：{error}"
        )


# ============================================================
# 12. 执行
# ============================================================

def main() -> None:
    print("=" * 60)
    print("高饱和蓝色海洋图片筛选")
    print("=" * 60)

    print(
        f"目标品牌色：{TARGET_COLOR}"
    )

    print(
        "目标 HSV："
        f"H={TARGET_HUE:.3f}, "
        f"S={TARGET_SATURATION:.3f}, "
        f"V={TARGET_VALUE:.3f}"
    )

    print(
        f"目标图片数量：{TARGET_COUNT}"
    )

    print(
        f"最大候选图片数量：{MAX_CANDIDATES}"
    )

    candidates = search_candidates()

    if not candidates:
        print(
            "\n没有搜索到候选图片。"
        )
        return

    print(
        f"\n候选图片搜索完成："
        f"{len(candidates)} 张"
    )

    accepted = score_candidates(
        candidates
    )

    if not accepted:
        print(
            "\n没有图片通过筛选。"
            "\n建议依次尝试："
            "\n1. 将 MIN_BLUE_RATIO 改为 0.50"
            "\n2. 将 MIN_HIGH_SATURATION_RATIO 改为 0.45"
            "\n3. 将 MIN_CORE_BLUE_RATIO 改为 0.15"
            "\n4. 将 MAX_CANDIDATES 提高到 800"
        )
        return

    if len(accepted) < TARGET_COUNT:
        print(
            f"\n注意：只筛选出 {len(accepted)} 张，"
            f"不足目标数量 {TARGET_COUNT} 张。"
        )

    saved_results = download_final_images(
        accepted
    )

    save_score_csv(
        saved_results
    )

    print("\n" + "=" * 60)
    print("下载完成")
    print("=" * 60)

    print(
        f"最终保存图片："
        f"{len(saved_results)} 张"
    )

    print(
        f"图片文件夹："
        f"{SAVE_FOLDER.resolve()}"
    )

    print(
        f"评分表："
        f"{RESULT_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()