import os
import requests


ACCESS_KEY = "nsgVm__RC25IJUqFpRuwlvGTriH7HFzOCuLdsTCPQV4"

TARGET_COLOR = "#105CF4"

keywords = [
    "deep blue ocean",
    "royal blue ocean",
    "vivid blue sea",
    "bright blue ocean water",
    "cobalt blue sea",
    "blue ocean waves",
    "open deep blue ocean",
    "clear blue sea surface"
]

save_folder = "images"
target_count = 100

os.makedirs(save_folder, exist_ok=True)


def download_image(url: str, filename: str) -> bool:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(filename, "wb") as file:
            file.write(response.content)

        return True

    except requests.RequestException as error:
        print(f"Download failed: {error}")
        return False


count = 1
downloaded_photo_ids = set()

for keyword in keywords:
    if count > target_count:
        break

    print(f"\nSearching: {keyword}")

    page = 1

    while count <= target_count:
        search_url = "https://api.unsplash.com/search/photos"

        params = {
            "query": keyword,
            "page": page,
            "per_page": 30,
            "orientation": "landscape",
            "content_filter": "high"
        }

        headers = {
            "Authorization": f"Client-ID {ACCESS_KEY}"
        }

        try:
            response = requests.get(
                search_url,
                params=params,
                headers=headers,
                timeout=30
            )

            response.raise_for_status()
            data = response.json()

        except requests.RequestException as error:
            print(f"Search failed: {error}")
            break

        photos = data.get("results", [])

        if not photos:
            break

        for photo in photos:
            if count > target_count:
                break

            photo_id = photo["id"]

            # 防止同一张照片因为不同关键词被重复下载
            if photo_id in downloaded_photo_ids:
                continue

            downloaded_photo_ids.add(photo_id)

            image_url = photo["urls"]["regular"]

            filename = os.path.join(
                save_folder,
                f"sea_{count:03d}.jpg"
            )

            print(f"Downloading {filename}")

            success = download_image(
                image_url,
                filename
            )

            if success:
                count += 1

        page += 1

        # 避免某一个关键词翻太多页
        if page > 5:
            break


print("\nDownload completed.")
print(f"Target brand colour: {TARGET_COLOR}")
print(f"Downloaded images: {count - 1}")