from pathlib import Path
from queue import Queue
from threading import Thread

import requests
from PIL import Image


q = Queue(1000)
DATA_DIR = Path("./images_data")
CROP_DIR = Path("./images_data_crop")


def download(imgname, url):
    pstr = f"download image {imgname} ..."

    if url == "None":
        print(f"{pstr} None")
        return

    image_fullpath = DATA_DIR / imgname
    if image_fullpath.exists():
        print(f"{pstr} Exist, Skip!")
        return

    response = requests.get(url, stream=True)
    with image_fullpath.open("wb") as f:
        for chunk in response.iter_content(chunk_size=512):
            f.write(chunk)

    print(f"{pstr} Done!")


def crop(image_name, crop_xy):
    pstr = f"crop image {image_name} ..."
    crop_xy = list(map(int, crop_xy.split()))

    image_fullpath = DATA_DIR / image_name
    if image_fullpath.exists():
        cropped_image_fullpath = CROP_DIR / image_name

        with Image.open(image_fullpath) as img:
            crop_area = (crop_xy[2], crop_xy[0], crop_xy[3], crop_xy[1])
            img = img.crop(crop_area)
            img = img.resize((600, 800))
            img.save(cropped_image_fullpath)
        print(f"{pstr} Done!")
    else:
        print(f"{pstr} Does not exist!")


def thread_work():
    while True:
        job, params = q.get()

        if job == "download":
            download(*params)
        elif job == "crop":
            crop(*params)

        q.task_done()


def main():
    if not DATA_DIR.is_dir():
        DATA_DIR.mkdir()

    for _ in range(50):
        t = Thread(target=thread_work)
        t.daemon = True
        t.start()

    with open("alldata_urls.txt", "r") as f:
        image_infos = f.read().strip().split("\n")

    for image_info in image_infos:
        q.put(["download", image_info.split()])
    q.join()

    print("\t\tDownload - Done!")

    for image_info in image_infos:
        image_name = image_info.split()[0]

        image_fullpath = DATA_DIR / image_name
        if not image_fullpath.exists():
            continue

        filesize = image_fullpath.stat().st_size
        if filesize <= 10 * 1024:
            print(f"Remove {image_name} which size {filesize} bytes below 10 * 1024 bytes")
            image_fullpath.unlink()

    # crop the downloaded images
    if not CROP_DIR.is_dir():
        CROP_DIR.mkdir()

    with open("crop.txt", "r") as f:
        crop_infos = f.read().strip().split("\n")

    for crop_info in crop_infos:
        q.put(["crop", crop_info.split(maxsplit=1)])
    q.join()

    print("\t\tEverything - Done!")


if __name__ == "__main__":
    main()
