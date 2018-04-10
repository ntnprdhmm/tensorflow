import urllib.request
from tqdm import tqdm
import os

# on URL per line
FILENAME = 'urls.txt'
IMAGE_FOLDER = 'images'

for url in tqdm(open(FILENAME, 'r').readlines()):
    url = url.strip()
    image_name = url.split('/')[-1]
    IMAGE_PATH = IMAGE_FOLDER + '/' + image_name
    if not os.path.exists(IMAGE_FOLDER + '/' + image_name):
        try:
            urllib.request.urlretrieve(url, IMAGE_PATH)
        except:
            pass
