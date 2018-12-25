import os
import imageio
import cv2
import numpy as np
from keras.utils import Sequence


# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class trainSequence(Sequence):

    def __init__(self, batch_size):
        self.x = os.listdir('train/color')
        self.y = os.listdir('train/normal')
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_masks = [imageio.imread(os.path.join('train/mask',file_name)) for file_name in batch_x]
        #print(batch_masks.shape)


        out_x =  np.array([imageio.imread(os.path.join('train/color',file_name)) for file_name in batch_x]).astype('float32')
        out_y =  [imageio.imread(os.path.join('train/normal',file_name)) for file_name in batch_y]
        out_y =  np.array([cv2.bitwise_and(y,y,mask = mask) for y,mask in zip(out_y,batch_masks)]).astype('float32')

        return out_x, out_y