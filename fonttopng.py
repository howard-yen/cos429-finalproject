from os import path, mkdir

import numpy as np
import matplotlib.pyplot as plt
from freetype import *

def save_fonts(ttf_path, idx, save_dir, font_size=48, img_size=128):
    if not path.exists(save_dir):
        print('not exist')
        mkdir(save_dir)

    face = Face(ttf_path)
    face.set_pixel_sizes(font_size, font_size)
    slot = face.glyph

    #alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    save_files = []
    for i, c in enumerate(alphabet):
        face.load_char(c)
        bitmap = slot.bitmap

        h = bitmap.rows
        w = bitmap.width
        
        x = (img_size - h) // 2
        y = (img_size - w) // 2
        
        if h > 64 or w > 64:
#             print(h, w)
            return -1

#         if h < 20 or w < 20:
#             print(h, w)

        if not path.exists(path.join(save_dir, f'{c}')):
            mkdir(path.join(save_dir, f'{c}'))

        imgpath = path.join(save_dir, f'{c}', f'{idx}')
        img = np.zeros((img_size, img_size, 1), dtype='uint8')

        temp = np.reshape(bitmap.buffer[:h*w], (h, w))
        try:
            img[x:x+h, y:y+w, 0] = temp
        except ValueError:
            os.remove(ttf_path)
            print(ttf_path)
            return -2

        #plt.imsave(imgpath, img, cmap='gray')

        save_files.append((imgpath, img))
#         if w == 0 or h == 0:
#             print(ttf_path)
#             plt.imshow(img, cmap='gray')
#             plt.show()

    for (impath, img) in save_files:
        np.save(impath, img)

    return 0

def main():
    count = 0
    c2 = 0
    f = open('fonts.csv', 'w')
    idx = 0
    for i, entry in enumerate(os.scandir('./fonts')):
        if entry.is_file():
            ret = save_fonts(entry.path, idx, './images')
            if ret == -1:
                count+=1
            elif ret == -2:
                c2 += 1
            else:
                idx += 1
                f.write(f'{idx} {entry.path}\n')

    f.close()
    print(f'count is {count}')
    print(f'c2 is {c2}')

if __name__=='__main__':
    main()
