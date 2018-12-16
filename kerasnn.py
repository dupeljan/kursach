from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import os
import cv2

TEST_PATH = "filtred5"
DATASET_PATH = TEST_PATH +"_dataset" + ".npz"

def create_dataset():
	images = list()
	labels = list()
	files = os.listdir(TEST_PATH)
	for filename in files:
		name,ext = os.path.splitext(filename)
		if ext == ".png":
			img = cv2.imread(os.path.join(TEST_PATH,name+'.png'))
			img = np.compress([True],cv2.resize(img, (28, 28)).reshape(784,3), axis=1 )
			images.append( img )
			labels.append( int(name[0]) )
	np.savez(DATASET_PATH,set= np.array( [images, labels] ) )
	
def get_dataset(path= DATASET_PATH):
	data = np.load(path)['set']
	return ( np.stack(data[0],axis=0) , np.stack(data[1],axis=0) )


def create_network():
	#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	images,labels = get_dataset()
	print(images.shape)
	train_images, test_images = images[:1200], images[1200:]
	train_labels, test_labels = labels[:1200], labels[1200:]
	network = models.Sequential()
	network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
	network.add(layers.Dense(10, activation='softmax'))

	network.compile(optimizer='rmsprop',
	loss='categorical_crossentropy',
	metrics=['accuracy'])


	train_images = train_images.reshape((1200, 28 * 28))
	train_images = train_images.astype('float32') / 255
	test_images = test_images.reshape((200, 28 * 28))
	test_images = test_images.astype('float32') / 255

	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)

	network.fit(train_images, train_labels, epochs=5, batch_size=128)	

def main():
	create_network()

if __name__ == '__main__':
	main()
