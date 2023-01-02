import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import random
import datetime
import poker_tools
from tools import CARD_VECTOR_DICT 

"""
Here we'll define the player class
"""

class goodness_predictor():
	"""A class to hold the goodness predictor
	"""
	def __init__(self):
		self.graph = tf.Graph()

		#construct graph
		with self.graph.as_default():
			self.x1_ = tf.placeholder(
				dtype = tf.float32,
				shape = [None, 105], 
				name = "intputs"
				)

			#Make network

			self.layer_1 = tf.layers.Dense(
				units = 1024,
				activation = tf.nn.relu 
				)

			self.hidden_1 = tf.layers.Dense(
				units = 1024,
				activation = tf.nn.relu 
				)

			self.layer_2 = tf.layers.Dense(
				units = 1,
				activation = None 
				)

			self.n1_layer_1 = self.layer_1(self.x1_)
			self.n1_hidden_1 = self.hidden_1(self.n1_layer_1)
			self.G1 = self.layer_2(self.n1_layer_1)

			self.saver = tf.train.Saver()

	def restore(self, session):
		self.saver.restore(session, "/updated_tmp_goodness_hidden_1/model_most_recent.ckpt")

	def goodness(self, session, vectors = None, holes = None, communities = None):
		if type(vectors) != type(None):
			return session.run(-self.G1, feed_dict = {self.x1_ : vectors})
		else:		
			return session.run(-self.G1, feed_dict = {self.x1_ : self.vectorize_hands(holes, communities)})
		"""
		if type(vectors) != type(None):
			with session as sess:
				self.saver.restore(sess, "/updated_tmp_goodness_hidden_1/model_most_recent.ckpt")

				g1 = sess.run(-self.G1, feed_dict = {self.x1_ : vectors})
				return g1
		else:		
			with session as sess:
				self.saver.restore(sess, "/updated_tmp_goodness_hidden_1/model_most_recent.ckpt")

				g1 = sess.run(-self.G1, feed_dict = {self.x1_ : self.vectorize_hands(holes, communities)})
				return g1"""

	def vectorize_hands(self, holes, communities, normalize = False):
		#takes (hole1, hole2, community)
		#outputs corresponding 53 dim vectors
		assert(len(holes) == len(communities))
		vectors = []

		for i, hole in enumerate(holes):
			community = communities[i]

			v= np.zeros(105)

			for c in community:
				v[52+CARD_VECTOR_DICT[c]]+=1

			for _ in range(5-len(community)):
				v[104]+=1

			for c in hole:
				v[CARD_VECTOR_DICT[c]]+=1

			if normalize:
				v -= np.mean(v1)
				v /= np.std(v1)

			vectors.append(v)

		return np.array(vectors)
"""
print('gtest')
g = goodness_predictor()


print(g.goodness(tf.Session(graph = g.graph), holes = [['3s', '2c']], communities = [[]]))

print(g.goodness(tf.Session(graph = g.graph), holes = [['As', 'Ac']], communities = [[]]))
"""
class volatility_predictor():
	"""A class to hold the volatility predictor
	"""
	def __init__(self):
		self.graph = tf.Graph()

		#construct graph
		with self.graph.as_default():
			self.card_vector_ = tf.placeholder(
				dtype = tf.float32,
				shape = [None, 105], 
				name = "inputs"
				)

			#Make network


			self.vol_layer_1 = tf.layers.Dense(
				units = 1024,
				activation = tf.nn.relu 
				)

			self.hidden_1 = tf.layers.Dense(
				units = 1024,
				activation = tf.nn.relu 
				)

			self.vol_layer_2 = tf.layers.Dense(
				units = 1,
				activation = None 
				)

			self.vol_net_layer_1 = self.vol_layer_1(self.card_vector_)
			vol_net_hidden_1 = self.hidden_1(self.vol_net_layer_1)
			self.volatilities = self.vol_layer_2(self.vol_net_layer_1)

			self.saver = tf.train.Saver()

	def restore(self, session):
		"""restore old session"""
		self.saver.restore(session, "/volatility_0/model_most_recent.ckpt")


	def volatility(self, session, vectors = None, holes = None, communities = None):
		if type(vectors) != type(None):
			return session.run(self.volatilities, feed_dict = {self.card_vector_ : vectors})
		else:		
			return session.run(self.volatilities, feed_dict = {self.card_vector_ : self.vectorize_hands(holes, communities)})
		"""
		if type(vectors) != type(None):
			with session as sess:
				self.saver.restore(sess, "/volatility_0/model_most_recent.ckpt")

				v = sess.run(self.volatilities, feed_dict = {self.card_vector_ : vectors})
				return v
		else:		
			with session as sess:
				self.saver.restore(sess, "/volatility_0/model_most_recent.ckpt")

				v = sess.run(self.volatilities, feed_dict = {self.card_vector_ : self.vectorize_hands(holes, communities)})
				return v"""

	def vectorize_hands(self, holes, communities, normalize = False):
		#takes (hole1, hole2, community)
		#outputs corresponding 53 dim vectors
		assert(len(holes) == len(communities))
		vectors = []

		for i, hole in enumerate(holes):
			community = communities[i]

			v= np.zeros(105)

			for c in community:
				v[52+CARD_VECTOR_DICT[c]]+=1

			for _ in range(5-len(community)):
				v[104]+=1

			for c in hole:
				v[CARD_VECTOR_DICT[c]]+=1

			if normalize:
				v -= np.mean(v1)
				v /= np.std(v1)

			vectors.append(v)

		return np.array(vectors)
"""
r = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
s = ['s', 'd', 'c', 'h']

cards = [a+b for a in r for b in s]
holes = []
for i, c in enumerate(cards):
	for j in range(i):
		holes.append([cards[i], cards[j]])

print('vtest')
g = volatility_predictor()


vs=g.volatility(tf.Session(graph = g.graph), holes = holes, communities = [[] for _ in holes])


a = list(zip(vs, holes))
a.sort()

print(a)"""

def sum_gradients(gradient_list):
	"""a tool to combine the gradients
	we assume the gradients have the same shape
	"""
	#first filter list
	gradient_list = list(filter(lambda x: bool(x), gradient_list))
	if gradient_list == []:
		return 0


	out = gradient_list.pop()
	for g in gradient_list:
		for i, v in enumerate(g):
			out[i]+=v 

	return np.array(out)





class poker_player_v0():

	def __init__(self, name):
		"""Here we need to 
		name the player, 
		define the network,
		create a method to save it
		create a method to load it
		"""
		self.name = name
		self.hand_stats = [None, None] #goodness, volatility

		self.network()

	def init_global_vars(self, session):
		session.run(self.initializer)

	def test_dict(self, n):

		td = {
			self.g_ : 0.0,
			self.v_ : 0.0,
			self.money_ : 1,
			self.bet_by_me_ : 1,
			self.pot_total_ : 10, 
			self.raise_ : 10,
			self.round_num_ : 3,
			self.num_players_ : 3.,

			self.opponent_fold_ : [[0] for _ in range(n)],
			self.opponent_call_ : [[0] for _ in range(n)],
			self.opponent_last_move_ : [[1, 2, 3] for _ in range(n)],
			self.opponent_current_bets_ : [[0] for _ in range(n)],
			self.total_bet_by_opponent_ : [[0] for _ in range(n)],
			self.opponent_not_yet_played_ : [[0] for _ in range(n)]
		}
		return td

	def network(self):
		self.graph = tf.Graph()

		with self.graph.as_default():
			with tf.variable_scope(self.name):

				"""
				Standard inputs
				"""
				self.g_ = tf.placeholder(
					dtype = tf.float32,
					shape = [], 
					name = "hand_goodness"
					)

				self.v_ = tf.placeholder(
					dtype = tf.float32,
					shape = [], 
					name = "hand_volatility"
					)

				self.money_ = tf.placeholder(
					dtype = tf.float32,
					shape = [], 
					name = "money"
					)

				self.bet_by_me_ = tf.placeholder(
					dtype = tf.float32,
					shape = [], 
					name = "bet_by_me"
					)

				self.pot_total_ = tf.placeholder(
					dtype = tf.float32,
					shape = [], 
					name = "pot"
					)

				self.raise_ = tf.placeholder(
					dtype = tf.float32,
					shape = [], 
					name = "raise_amount"
					)

				#0, 3, 4, 5
				self.round_num_ = tf.placeholder(
					dtype = tf.int32,
					shape = [], 
					name = "round_number"
					)

				self.num_players_ = tf.placeholder(
					dtype = tf.float32,
					shape = [], 
					name = "num_players"
					)

				"""
				Repeated inputs
				"""
				self.opponent_fold_ = tf.placeholder(
					dtype = tf.float32,
					shape = [None, 1], 
					name = "bet_by_opponent"
					)

				self.opponent_call_ = tf.placeholder(
					dtype = tf.float32,
					shape = [None, 1], 
					name = "bet_by_opponent"
					)

				self.opponent_last_move_ = tf.placeholder(
					dtype = tf.float32,
					shape = [None, 3], 
					name = "bet_by_opponent"
					)

				self.opponent_current_bets_ = tf.placeholder(
					dtype = tf.float32,
					shape = [None, 1], 
					name = "bet_by_opponent"
					)

				self.total_bet_by_opponent_ = tf.placeholder(
					dtype = tf.float32,
					shape = [None, 1], 
					name = "bet_by_opponent"
					)

				self.opponent_not_yet_played_ = tf.placeholder(
					dtype = tf.float32,
					shape = [None, 1], 
					name = "bet_by_opponent"
					)

				"""Combine into vectors
				"""

				self.standard_input_vector_ = tf.reshape(
					tf.concat(
						[[
							self.g_, 
							self.v_, 
							self.money_, 
							self.bet_by_me_, 
							self.pot_total_,
							self.raise_,
							self.num_players_], 
							tf.reshape(tf.one_hot(self.round_num_, 4), [4]),
						], 0
					), [-1, 11])

				self.player_input_vector_ = tf.concat(
					[self.opponent_fold_, 
					self.opponent_call_, 
					self.opponent_last_move_, 
					self.opponent_current_bets_, 
					self.total_bet_by_opponent_,
					self.opponent_not_yet_played_,
					tf.ones_like(self.opponent_not_yet_played_)*self.num_players_], -1
					)

				"""
				We make a layer that processes the players around the table 
				"""
				self.player_layer_0 = tf.layers.Dense(
					units = 256,
					activation = tf.nn.relu 
					)(self.player_input_vector_)

				#no activation on this layer, since I'll be combining it with another layer
				#then 'activating'
				self.player_layer_1 = tf.layers.Dense(
					units = 1024,
					activation = None
					)(self.player_layer_0)

				"""
				We make a layer to process the standard inputs
				"""
				self.standard_input_layer = tf.layers.Dense(
					units = 1024,
					activation = None 
					)(self.standard_input_vector_)

				#lets try this
				self.combination_layer = tf.nn.relu(self.standard_input_layer + tf.reduce_mean(self.player_layer_1, axis = 0))

				self.layer_1 = tf.layers.Dense(
					units = 1024,
					activation = tf.nn.relu
					)(self.combination_layer)

				self.layer_2 = tf.layers.Dense(
					units = 1024,
					activation = tf.nn.relu
					)(self.layer_1)

				self.nn_output = tf.layers.Dense(
					units = 4,
					activation = None
					)(self.layer_2)

				#outputs
				self.cfr_logit, self.raise_amount_unscaled = tf.split(self.nn_output, [3,1], 1)
				self.cfr_log_probs = tf.nn.log_softmax(self.cfr_logit)

				#log probs
				self.c_log_probs, self.f_log_probs, self.r_log_probs = tf.split(self.cfr_log_probs, [1,1,1], 1)

				#raise amount
				self.raise_amount = tf.sigmoid(self.raise_amount_unscaled) #just for now we limit bets

			#derivatives
			self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

			self.c_derivs = tf.gradients(self.c_log_probs, self.vars)
			self.f_derivs = tf.gradients(self.f_log_probs, self.vars)
			self.r_derivs = tf.gradients(self.r_log_probs, self.vars)

			self.raise_derivs = tf.gradients(self.raise_amount, self.vars)

			#self.grad_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, None])
			#self.grad_placeholder = tf.placeholder(dtype = tf.float32, shape = [None])

			self.placeholder_gradients = []
			for i, var in enumerate(self.vars):
				self.placeholder_gradients.append((tf.placeholder(tf.float32, shape= var.get_shape(), name = 'var_'+str(i)), var))
				

			dx = 0.01
			self.train_op = tf.train.GradientDescentOptimizer(dx).apply_gradients(self.placeholder_gradients)

			#initializer
			self.initializer = tf.global_variables_initializer()
			self.saver = tf.train.Saver()


	def make_feed_dict(self, state_dict):
		"""state is the dictionary returned by the game class
		assumes hand stats have already been determined by the goodness and volitility nns
		"""
		d = {
			self.g_ : self.hand_stats[0],
			self.v_ : self.hand_stats[1],
			self.money_ : state_dict['money'],
			self.bet_by_me_ : state_dict['bet_by_me'],
			self.pot_total_ : state_dict['pot'], 
			self.raise_ : state_dict['raise'],
			self.round_num_ : state_dict['round_num'],
			self.num_players_ : state_dict['player_num'],

			self.opponent_fold_ : state_dict['fold_bools'],
			self.opponent_call_ : state_dict['call_bools'],
			self.opponent_last_move_ : state_dict['last_move'],
			self.opponent_current_bets_ : state_dict['current_bets'],
			self.total_bet_by_opponent_ : state_dict['total_bets'],
			self.opponent_not_yet_played_ : state_dict['not_yet_played']
		}

		return d

	def train_step(self, gradients, session, dx = 0.01):
		"""Returns training op to run

		decision_gradients takes a list of gradients: the c, f, and r gradients

		allows for some 
		"""

		feed_dict = {}



		for i, D in enumerate(gradients):
			if type(D)!=type(None):
				feed_dict[self.placeholder_gradients[i][0]] = D
			else:
				feed_dict[self.placeholder_gradients[i][0]] = np.zeros(shape = self.vars[i].get_shape())

		session.run(self.train_op, feed_dict = feed_dict)


		#grads_and_vars = list(zip(gradients, self.vars))
		#make this better
		#dx=0.01
		#with self.graph.as_default(): 
		#	train = tf.train.GradientDescentOptimizer(dx).apply_gradients(grads_and_vars)
		
		#session.run(train, feed_dict = {self.num_players_ : num_players})

		#session.run(self.train_op, feed_dict = {self.grads_and_vars_placeholder : grads_and_vars})
		#session.run(self.train_op.apply_gradients(grads_and_vars))

	def save(self, session):
		"""save the session
		"""
		self.saver.save(session, "/player_folder/"+self.name+"model_most_recent.ckpt")

	def load(self, session):
		"""Restore the session
		"""
		self.saver.restore(session, "/player_folder/"+self.name+"model_most_recent.ckpt")


game = poker_tools.poker_game(10)

def random_move(x = None):
	if x== None:
		x = np.random.rand()*3
		if x<1:
			return [1, 0, 0, 0, [0], [0]]
		elif x<2:
			return [0, 1, 0, 0, [0], [0]]
		else:
			return [0, 0, 1, 1, [0], [0]]

	else:
		if x<1:
			return [1, 0, 0, 0, [0], [0]]
		elif x<2:
			return [0, 1, 0, 0, [0], [0]]
		else:
			return [0, 0, 1, 1, [0], [0]]

"""
x = poker_player_v0('throwaway1')

graphs = [tf.Graph()]


sess = tf.Session(graph = x.graph)
x.init_global_vars(sess)


y = sess.run(x.nn_output, feed_dict = x.test_dict(2))

print(y.shape)
y = sess.run(x.player_input_vector_, feed_dict = x.test_dict(2))

print(y.shape)
print(y)

"""
