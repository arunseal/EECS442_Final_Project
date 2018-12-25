import os
import shutil 

val_dir = 'validation'
train_dir = 'actual_training'
root_dir = 'train'

list_of_dirs = ['color','mask','normal']


if not os.path.exists(val_dir):
	os.mkdir(val_dir)

if not os.path.exists(train_dir):
	os.mkdir(train_dir)

for dirs in list_of_dirs:
	file_path_val = os.path.join(val_dir,dirs)
	os.mkdir(file_path_val)
	file_path_train = os.path.join(train_dir,dirs)
	os.mkdir(file_path_train)
	file_path_root = os.path.join(root_dir,dirs)
	filenum = 0
	for root , folders, files in os.walk(file_path_root):
		for file in files:
			file_path = os.path.join(file_path_root,file)
			if (filenum % 10 == 0):
				shutil.copy(file_path,file_path_val)
			else:
				shutil.copy(file_path,file_path_train)
			filenum += 1
		

	









