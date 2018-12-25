import imageio
import numpy as np
import os.path

dirs = ['color/', 'mask/', 'normal/']
for dir in dirs:
    for i in range(20000):
        pth = dir + str(i) + '.png'
        if(os.path.isfile(pth)):
            im = imageio.imread(pth)
            im = np.flip(im, axis=1)
            num = 20000 + i
            imageio.imwrite(dir + str(num) + '.png', im)