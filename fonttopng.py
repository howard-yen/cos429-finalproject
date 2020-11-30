from os import path, mkdir

import numpy as np
import matplotlib.pyplot as plt
from freetype import *

def save_fonts(ttf_path, idx, save_dir, font_size=24, img_size=48):
    if not path.exists(save_dir):
        mkdir(save_dir)

    print(ttf_path)
    face = Face(ttf_path)
    face.set_pixel_sizes(font_size, font_size)
    slot = face.glyph

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i, c in enumerate(alphabet):
        face.load_char(c)
        bitmap = slot.bitmap

        h = bitmap.rows
        w = bitmap.width

        x = (img_size - h) // 2
        y = (img_size - w) // 2

        imgpath = os.path.join(save_dir, f'{c}', f'{idx}.png')
        img = np.zeros((img_size, img_size, 1), dtype='uint8')

        temp = np.reshape(bitmap.buffer[:h*w], (h, w,1))
        print(c)
        print(img.shape, temp.shape)
        print(h, w, x, y)
        img[x:x+h, y:y+w] = temp

        #plt.imsave(imgpath, img, cmap='gray')
        np.save(imgpath, img)

def main():
    f = open('fonts.csv', 'w')
    for i, entry in enumerate(os.scandir('./fonts')):
        if entry.is_file():
            f.write(f'{i}, {entry.path}\n')
            save_fonts(entry.path, i, './images')

    f.close()

if __name__ == '__main__':
    main()
