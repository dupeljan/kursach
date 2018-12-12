import cv2
import numpy as np
from random import random
import os

SOURCE_PATCH = "test"
RESULT_PATH = "noise test"

def add_noise(name):
	k = 35#20
	img = cv2.imread(os.path.join(SOURCE_PATCH ,  name ))

	for i in range(len(img)):
		for j , x in enumerate(img[i]):
			img[i][j] = x + x*[(random() - 0.5)/k if x[0] != 255 else 1,\
							   (random() - 0.5)/k if x[1] != 255 else 1,\
							   (random() - 0.5)/k if x[2] != 255 else 1]  		
	cv2.imwrite(os.path.join(RESULT_PATH , name+" noisy"+".png"),img)

def main():
	if not os.path.isdir(RESULT_PATH):
		os.mkdir(RESULT_PATH)
	files = os.listdir(SOURCE_PATCH)
	count = 0
	for filename in files:
		name,ext = os.path.splitext(filename)
		if ext == "":
			add_noise(name)
			count += 1
			print("not more than " + str(len(files) - count) + " files left" )
	print("Generate test success")
if __name__ == '__main__':
	main()