import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

SOURCE_PATCH = "test6"
RESULT_PATH = "filtred6"
LOG_FILE = "errors " + SOURCE_PATCH


def filter(name):
	#name = '1'
	rgb = cv2.imread(os.path.join(SOURCE_PATCH,name+'.png'))

	lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2Lab)

	average_a = np.array( list(x[1]  for i in range(len(lab)) for x in lab[i] )) .mean()
	average_b = 0
	
	a = cv2.split(lab)[1]
	if average_a < 130: # МНОГО ЗЕЛЕНОГО ИЛИ СИНЕГО, В А КОНТРАСТ 
		rgb = cv2.threshold(a,132,255,cv2.THRESH_BINARY)[1]
		state = 1
	else:
		average_b = np.array( list(x[2]  for i in range(len(lab)) for x in lab[i] )) .mean()
		if  average_b < 135: #130 134 # МАЛО КРАСНОГО => КРАСНЫЙ КАНАЛ ТЕМНЫЙ, В А КОНТРАСТ
			state = 2
			rgb = cv2.threshold(a,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]#'''average'''
		else:
			state = 3 # МНОГО КРАСНОГО => КРАСНЫЙ КАНАЛ СВЕТЛЫЙ, В КРАСНОМ КОНТРАСТ
			rgb = cv2.threshold(cv2.split(rgb)[2],227,255,cv2.THRESH_BINARY)[1] # Red chanel
			rgb = cv2.bitwise_not(rgb)
	error = False
	if ( name[-1] != str(state)):
		error = True
		file = open(LOG_FILE, 'a' )
		file.write("a "+ str(average_a) + " b " + str(average_b) + " " + name + "\n"\
					"Got type " + str(state) +"\n")
		file.close()
	
	kernel_opening = np.ones((2,2),np.uint8) # 2
	kernel_closing = np.ones((12, 12),np.uint8)#30

	opening = cv2.morphologyEx(rgb, cv2.MORPH_OPEN, kernel_opening)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_closing)

	dst = cv2.medianBlur(closing,5)

	cv2.imwrite( os.path.join(RESULT_PATH ,  name +".png"),dst )
	print(name+".png filtered")

	return error

def main():
	if os.path.isfile(LOG_FILE):
		os.remove(LOG_FILE)
	if not os.path.isdir(RESULT_PATH):
		os.mkdir(RESULT_PATH)
	files = os.listdir(SOURCE_PATCH)
	errors = 0
	for filename in files:
		name,ext = os.path.splitext(filename)
		if ext == ".png":
			if (filter(name)):
				errors += 1
	file = open(LOG_FILE, 'a' )
	file.write("Error counts: " + str(errors) + '\n')
	file.write("% " + str(errors / float(len(files)) ) + '\n')
	file.write("#"*40)
	file.close()

if __name__ == '__main__':
	main()