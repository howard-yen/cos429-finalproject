from os import path, mkdir

import numpy as np
import matplotlib.pyplot as plt
from freetype import *

def save_fonts(ttf_path, idx, save_dir, font_size=32):
    if not path.exists(save_dir):
        mkdir(save_dir)

    face = Face(ttf_path)
    face.set_pixel_sizes(font_size, font_size)
    slot = face.glyph

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i, c in enumerate(alphabet):
        face.load_char(c)
        bitmap = slot.bitmap

        h = bitmap.rows
        w = bitmap.width

        x = (font_size - h) // 2
        y = (font_size - w) // 2

        imgpath = os.path.join(save_dir, f'{c}', f'{idx}.png')
        img = np.zeros((font_size, font_size), dtype='uint8')

        img[x:x+h, y:y+w] = np.reshape(bitmap.buffer, (h, w))

        plt.imsave(imgpath, img, cmap='gray')

def main():
    for i, entry in enumerate(os.scandir('./fonts')):
        if entry.is_file():
            with open('fonts.csv', 'a') as f:
                f.write(f'{i}, {entry.path}\n')
            save_fonts(entry.path, i, './images')

if __name__ == '__main__':
    main()
