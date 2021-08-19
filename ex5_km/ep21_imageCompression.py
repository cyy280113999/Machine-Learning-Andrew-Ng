"""
图片压缩
"""

import numpy as np
from PIL import Image
from ep2_km import kMeans

def imageCompression(img):
    pass

def main():
    raw_img = Image.open('冬-ab.jpg')
    raw_img = raw_img.resize((800, 480))
    # raw_img.show()
    img = np.array(raw_img,dtype=np.float)/255.0
    img = img.reshape(img.shape[0]*img.shape[1], 3)
    k = 16
    label, center, cvg = kMeans(img, k, iterations_max=10)
    print('converged?:', cvg)
    img = center[label]
    img = img.reshape(raw_img.size[1],raw_img.size[0],3)
    compressed_img = Image.fromarray((255*img).astype('uint8'))
    compressed_img.show()


if __name__ == '__main__':
    main()