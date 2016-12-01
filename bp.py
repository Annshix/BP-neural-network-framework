#coding:utf-8
import math
import random
import string

random.seed(0)

def rand(a,b):
	return (b-a)*random.random() + a

def makeMatrix(I,J,fill = 0.0):
	m = []
	for i in range(I):
		m.append([fill] * J)
	return m

def sigmoid(x):
	return math.tanh(x)

def dsigmoid(y):
	return 1.0 - y**2

class NN:
	'''three layers'''
	def __init__(self,ni,nh,no):
		self.ni = ni + 1
		self.nh = nh
		self.no = no

		self.ai = [1.0]*self.ni
		self.ah = [1.0]*self.nh
		self.ao = [1.0]*self.no

		self.wi = makeMatrix(self.ni, self.nh)
		self.wo = makeMatrix(self.nh, self.no)

		for i in range(self.ni):
			for j in range(self.nh):
				self.wi[i][j] = rand(-0.2, 0.2)
		for j in range(self.nh):
			for k in range(self.no):
				self.wo[j][k] = rand(-2.0, 2.0)

		self.ci = makeMatrix(self.ni, self.nh)
		self.co = makeMatrix(self.nh, self.no)

	def update(self, inputs):
		if len(inputs) != self.ni - 1:
			raise ValueError("Wrong number of nodes!")

		for i in range(self.ni - 1):
			'''self.ai[i] = sigmoid(inputs[i])'''
			self.ai[i] = inputs[i]

		for j in range(self.nh):
			sum = 0.0
			for i in range(self.ni):
				sum = sum + self.ai[i] * self.wi[i][j]
			self.ah[j] = sigmoid(sum)

		for k in range(self.no):
			sum = 0.0
			for j in range(self.nh):
				sum = sum + self.ah[j] * self.wo[j][k]
			self.ao[k] = sigmoid(sum)

		return self.ao[:]

	def backPropagate(self, targets, N, M):
		if len(targets) != self.no:
			raise ValueError("Wrong number of nodes!")

		output_deltas = [0.0] * self.no
		for k in range(self.no):
			error = targets[k] - self.ao[k]
			output_deltas[k] = dsigmoid(self.ao[k]) * error

		hidden_deltas = [0.0] * self.nh
		for j in range(self.nh):
			error = 0.0
			for k in range(self.no):
				error = error + output_deltas[k] * self.wo[j][k]
			hidden_deltas[j] = dsigmoid(self.ah[j]) * error

		for j in range(self.nh):
			for k in range(self.no):
				change = output_deltas[k] * self.ah[j]
				self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
				self.co[j][k] = change

		for i in range(self.ni):
			for j in range(self.nh):
				change = hidden_deltas[j] * self.ai[i]
				self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
				self.ci[i][j] = change

		error = 0.0
		for k in range(len(targets)):
			error = error + 0.5 * ((targets[k] - self.ao[k])** 2)
		return error

	def test(self,patterns):
		res = [0.0]*self.no
		i = 0
		for p in patterns:
			res[i] = self.update(p[0])
			print(p[0], '->', self.update(p[0]))
			i = i+1
		return res


	def weights(self):
		print("wi:")
		for i in range(self.ni):
			print(self.wi[i])
		print()
		print("wo:")
		for j in range(self.nh):
			print(self.wo[j])

	def train(self, patterns, iterations = 1000, N = 0.5, M = 0.1):
		'''N: learning rate
		   M: momentum factor'''
		for i in range(iterations):
			error = 0.0
			for p in patterns:
				inputs = p[0]
				targets = p[1]
				self.update(inputs)
				error = error + self.backPropagate(targets,N,M)
			if i % 100 == 0:
				print("error: %-.5f" % error)

def demo():
	t_train = [
	
	]

	t_test = [[[1.0]]]

	m = NN(1, 3, 5)
	m.train(t_train)
	res = m.test(t_test)
	res = res[0]

	data_train = [
	
	]
	for d in data_train:
		# data here 
	data_test = [[res]]

	n = NN(5,5,1)

	n.train(data_train)

	n.test(data_test)
］

if __name__ == '__main__':
	demo()


