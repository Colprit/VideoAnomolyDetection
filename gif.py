from os import listdir, makedirs
from os.path import join, exists
from PIL import Image

from config import *

GIF_PATH = f'{TEST_PATH}/animated'

if not exists(GIF_PATH):
    makedirs(GIF_PATH)

def gen_test_gif(test_num):

    test_path = SINGLE_TEST_PATH.format(test_num)
    
    imgs = [
        Image.open(join(test_path, file))
        for file in sorted(listdir(test_path))
        if file.endswith('tif')
    ]

    gif_path = f'{GIF_PATH}/Test{test_num:03d}.gif'

    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=100,
        loop=0
    )

    return gif_path

if __name__=="__main__":
    for i in range(1, 10):
        gen_test_gif(i)