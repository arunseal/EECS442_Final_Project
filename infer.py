import sys
import os
import shutil
import imageio
import keras
from keras.models import load_model
import numpy as np


def get_data(data_dir):
	images = []
	fnames = os.listdir(data_dir)
	fnames = [x.strip('.png') for x in fnames]
	fnames = sorted(fnames, key = lambda x: int(x))

	for fname in fnames:
		image = imageio.imread(os.path.join(data_dir,fname+'.png'))
		image = np.array(image)
		images.append(image)
	images = np.array(images).astype('float32')
	return images



def main():
	if len(sys.argv) != 4:
		print('incorrect number of arguments')
		print('usage: python3 infer.py path_to_model path_to_data path_to_output')

	model_name = sys.argv[1]
	data_dir = sys.argv[2]
	pred_dir = sys.argv[3]

	shutil.rmtree(pred_dir)
	os.makedirs(pred_dir)

	X = get_data(data_dir)

	model = load_model(model_name)

	pred = model.predict(X)

	for idx, img in enumerate(pred):
		img = img.astype('uint8')
		fname = str(idx)+'.png'
		imageio.imwrite(os.path.join(pred_dir,fname), img)



if __name__ == '__main__':
	main()