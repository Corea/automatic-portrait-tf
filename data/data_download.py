import os
import urllib.request
from PIL import Image
from threading import Thread
from queue import Queue


q = Queue(1000)
data_folder = 'images_data'


def download(imgname, url):
    pstr = 'download image %s ...' % imgname

    if url == 'None':
        print(pstr + ' None')
        return

    imgpath = os.path.join(data_folder, imgname)
    if os.path.exists(imgpath):
        print(pstr + ' Exist, Skip!')
        return

    urllib.request.urlretrieve(url, imgpath)
    print(pstr + ' Done!')


def crop(imgname, crop_xy):
    pstr = 'crop image %s ...' % imgname
    crop_xy = list(map(int, crop_xy.split()))

    imgpath = os.path.join(data_folder, imgname)
    if os.path.exists(imgpath):
        img = Image.open(imgpath)
        area = (crop_xy[2], crop_xy[0], crop_xy[3], crop_xy[1])
        img = img.crop(area)
        img = img.resize((600, 800))
        img.save(os.path.join(data_folder + '_crop', imgname))
        print(pstr + ' Done!')
    else:
        print(pstr + ' Does not exist!')


def thread_work():
    while True:
        job, params = q.get()
        if job == 'download':
            download(*params)
        elif job == 'crop':
            crop(*params)
        q.task_done()


def main():
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    if not os.path.isdir(data_folder + '_crop'):
        os.mkdir(data_folder + '_crop')

    for i in range(50):
        t = Thread(target=thread_work)
        t.daemon = True
        t.start()

    imgs_names = open('alldata_urls.txt', 'r').read().strip().split('\n')
    for item in imgs_names:
        q.put(['download', item.split()])
    q.join()
    print('\t\tDownload - Done!')

    # crop the downloaded images
    crop_params = open('crop.txt', 'r').read().strip().split('\n')
    for item in crop_params:
        q.put(['crop', item.split(maxsplit=1)])
    q.join()
    print('\t\tEverything - Done!')


if __name__ == '__main__':
    main()
