import numpy as np
import scipy.special
from math import *

class Neural_network:
	"""docstring for Neural_network"""
	def __init__(self, layers_shape, speed= 0.5, alpha= 1 ):
		# Learning speed
		self.speed = speed

		self.alpha = alpha

		self.shape = layers_shape

		self.activation_function = lambda x: 1 / ( 1 - exp( -2 * alpha * x ))
		# Tensor of output all neurons
		self.output = np.ndarray(shape= (len(layers_shape) , ) , dtype= list)
		# Init weight
		self.weights_list = list()
		for i in range(len(layers_shape)-1):
			self.weights_list.append( np.random.normal(0.0, pow(layers_shape[i], -0.5), (layers_shape[i], layers_shape[i+1])))

	def sign(self,input):
		return [ self.activation_function(x) for x in input ]
	
	def predict(self,input, i= 0):
		y = self.sign( np.dot(input,self.weights_list[i]) )
		if len(self.weights_list) == i + 1 :
			return y
		return self.predict(y,i+1)

	def get_output(self,input, i= 1):
		'''
		Return output matrix
		'''
		y = self.sign( np.dot(input,self.weights_list[i-1]) )
		self.output[i] = y 
		if len(self.output) == i + 1:
			self.output[0] = input
			return self.output
		return self.get_output(y,i+1)

	def lern(self,input,label):
		print(self.weights_list)
		output = self.get_output(input)
		delta =  output[1:]
		# DESCENT
		# LAST LAYER
		for i, x in enumerate(output[-1]):
			delta[-1][i] = -2 * self.alpha * x * ( 1 - x ) * ( label[i] - x )
		# HIDEN LAYERS
		# REVERSE, BEGIN FROM SECOND FROM END
		for i in range( len(delta) - 2 , -1, -1) : # CHOOSE LAYER
			for j, x in enumerate(output[i]): # CHOOSE NEURON
				delta[i][j] = 2 * self.alpha * ( 1 - x ) * sum ( delta[i+1][k] * self.weights_list[i+1][j][k] for k in range( len( delta[i+1] ) ) ) 
		# UPDATE WEIGHTS
		for layer in range(len(self.weights_list)): # CHOOSE LAYER
			for i in range(len(self.weights_list[layer])): # CHOOSE ROW 
				for j in range(len(self.weights_list[layer][i])): # CHOOSE COLUMN
					self.weights_list[layer][i][j] -= self.speed * delta[layer][j] * output[layer][i] 



def main():
	nn = Neural_network([5,2])
	nn.lern([1,0,1,0,1],[1,0])
	print ( nn.predict([1,0,1,0,1]) )

if __name__ == '__main__':
	main()


		