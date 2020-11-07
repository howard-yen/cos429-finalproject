from freetype import *
import numpy as np
import cv2
from os import path, mkdir

def save_fonts(ttf_path, save_dir, font_size=32):
    if not path.exists(save_dir):
        mkdir(save_dir)

    face = Face(ttf_path)
    face.set_pixel_sizes(font_size, font_size)
    slot = face.glyph

    alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    for i, c in enumerate(alphabet):
        print(f"processing chars: {c}")
        face.load_char(c)
        bitmap = slot.bitmap

        h = bitmap.rows
        w = bitmap.width

        x = (font_size - w) //2
        y = (font_size - h) //2

        imgpath = os.path.join(save_dir, f'{c}.png')
        img = np.zeros((font_size, font_size, 1), dtype='uint8')

        for i in range(h):
            for j in range(w):
                img[y+i][x+j] = bitmap.buffer[i*w + j]

        cv2.imwrite(imgpath, img)

if __name__ == '__main__':
    ttf_path = './Hack-Regular.ttf'
    save_dir = './images'
    save_fonts(ttf_path, save_dir)

