import imageio
import os
import numpy as np




for fname in os.listdir('test/color'):
	image = imageio.imread(os.path.join('test/normal',fname))

	image = np.array(image)

	print np.max(image[:,:,0])
	print np.max(image[:,:,1])
	print np.max(image[:,:,2])
	print ''