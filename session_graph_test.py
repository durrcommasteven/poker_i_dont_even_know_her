"""
Here I'm just making a platform for various neural networks to compete

"""
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 



class player_0():
	def __init__(self, name):
		"""Here we need to 
		name the player, 
		define the network,
		create a method to save it
		create a method to load it
		"""
		self.name = name
		self.INIT = False
		self.graph = tf.Graph()
		self.network()

	def init_session(self, load = None):
		"""Initialize session, 
		"""
		self.session = tf.Session(graph = self.graph)

		if not load:
			self.session.run(self.initializer)
		else:
			self.load()

		self.INIT = True

	def network(self):
		"""define the graph
		"""

		with self.graph.as_default():

			self.initializer = tf.global_variables_initializer()

			with tf.variable_scope(self.name):
				self.x_ = tf.placeholder(
					dtype = tf.float32,
					shape = [None, 2], 
					name = "input"
					)

				self.y_ = tf.placeholder(
					dtype = tf.float32,
					shape = [None, 2], 
					name = "output"
					)

				self.out = tf.layers.Dense(
					units = 2,
					activation = None,
					name = 'layer'
					)(self.x_)

			self.loss = tf.reduce_mean(0.5*(self.y_ - self.out)**2)

			self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

			print(self.vars)

			self.grads = tf.gradients(self.loss, self.vars)

			self.saver = tf.train.Saver()

			self.dx = tf.placeholder(
					dtype = tf.float32,
					shape = [], 
					name = "input"
					)

			#print(self.grads.__len__())

			self.grads_and_vars = list(zip(self.grads, self.vars))

			self.GradientDescentOptimizer_Op = tf.train.GradientDescentOptimizer(self.dx).apply_gradients(self.grads_and_vars)

	def gradient(self, data, labels):
		"""Returns gradient
		"""
		return self.session.run(self.grads, feed_dict = {self.x_ : data, self.y_ : labels})

	def train_op(self, gradients, variables, dx = 0.1):
		"""Returns training op to run
		"""
		grads_and_vars = list(zip(gradients, variables))
		return 

	def save(self):
		"""save the session
		"""
		self.saver.save(self.session, "/player_folder/"+self.name+"model_most_recent.ckpt")

	def load(self):
		"""Restore the session
		"""
		self.saver.restore(self.session, "/player_folder/"+self.name+"model_most_recent.ckpt")

	def close_session(self):
		"""close the current session
		"""
		self.session.close()

#make data
thetas_0 = np.linspace(0, 2*np.pi, 501)
thetas_1 = thetas_0[-1]+thetas_0[:-1]

data = np.array([[np.cos(theta), np.sin(theta)] for theta in thetas_0])
labels = np.array([[np.cos(theta), np.sin(theta)] for theta in thetas_1])
#labels = np.array([[0] for theta in thetas_1])


def data_labels(num):
	indices = np.random.randint(0, high = 500, size = num)
	d, l = np.take(data, indices, axis = 0), np.take(labels, indices, axis = 0)
	#print(l.shape)
	return d, l

data_labels(100)
#try training

loss = []

player = player_0('throwaway')

player.init_session()

for _ in range(100):
	
	d, l = data_labels(100)

	g = player.gradient(d, l)
	#print(g)

	player.session.run(player.train_op(g, player.vars))

	tloss = player.session.run(player.loss, feed_dict = {player.x_ : d, player.y_ : l})
	print(tloss)
	loss.append(tloss)

player.close_session()

print(loss)
plt.plot(loss)
plt.show()

