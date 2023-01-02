import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import random
"""
Here I'm going to make a simple, multi-step task and see if i can train it with tensorflow
"""

def high_hand(policy1, policy2, session):
	"""
	in this game, each player gets a number in [0, 1]
	the players then undergo three rounds of betting 
	The reward is the profit of each player. 

	policy1 and policy2 are networks 

	the outputs are the rewards: -profit

	inputs to the policy are:
	x_ : x1,
	money_ : p1_money,
	round_num_ : turn,
	bet_by_me_ : bet_by_1, 
	bet_by_opponent_ : bet_by_2,
	raise_ : raise_amount

	outputs:
	p1_fold, p1_raise, p1_raise_amount

	policy1 always starts first

	"""
	pot = 1 #start out with something in the pot
	p1_fold_bool = False
	p2_fold_bool = False

	p1_money = 0
	p2_money = 0

	bet_by_1 = 0
	bet_by_2 = 0

	x1 = np.random.rand()
	x2 = np.random.rand()

	raise_amount = 0.

	for turn in range(3):

		#p1 starts
		p1_fold, p1_raise, p1_raise_amount = session.run(
			policy1, 
			feed_dict = {
				x_ : x1,
				money_ : p1_money,
				round_num_ : turn,
				bet_by_me_ : bet_by_1, 
				bet_by_opponent_ : bet_by_2,
				raise_ : raise_amount})

		#fold?
		p1_fold_bool = np.random.choice([True, False], p = [p1_fold, 1-p1_fold])
		if p1_fold_bool:
			return p1_money, p2_money+pot

		#raise?
		p1_raise_bool = np.random.choice([True, False], p = [p1_raise, 1-p1_raise])
		if p1_raise_bool:
			bet_by_1 += p1_raise_amount
			pot += p1_raise_amount
			p1_money -= p1_raise_amount
			raise_amount = p1_raise_amount
		else:
			raise_amount = 0

		#p2 goes now
		p2_fold, p2_raise, p2_raise_amount = session.run(
			policy2, 
			feed_dict = {
				x_ : x2
				money_ : p2_money,
				round_num_ : turn,
				bet_by_me_ : bet_by_2, 
				bet_by_opponent_ : bet_by_1,
				raise_ : raise_amount})

		#fold?
		p2_fold_bool = np.random.choice([True, False], p = [p2_fold, 1-p2_fold])
		if p2_fold_bool:
			return p1_money+pot, p2_money
		else:
			bet_by_2+=raise_amount
			pot+=raise_amount
			p2_money-=raise_amount

		#raise?
		p2_raise_bool = np.random.choice([True, False], p = [p2_raise, 1-p2_raise])
		if p2_raise_bool:
			bet_by_2 += p2_raise_amount
			pot += p2_raise_amount
			p2_money -= p2_raise_amount
			raise_amount = p2_raise_amount
		else:
			raise_amount = 0

	#at the end we choose the winner
	if x1>x2:
		return p1_money+pot, p2_money
	else:
		return p1_money, p2_money+pot


def raise_by_1(policy1, policy2, session):
	"""here all i want is for both policies to fold

	"""

	pot = 1 

	p1_money = 0
	p2_money = 0

	bet_by_1 = 0
	bet_by_2 = 0

	x1 = np.random.rand()
	x2 = np.random.rand()

	raise_amount = 0.

	#p1 starts
	p1_fold_logit, p1_raise, p1_raise_amount = session.run(
		policy1, 
		feed_dict = {
			x_ : x1,
			money_ : p1_money,
			round_num_ : turn,
			bet_by_me_ : bet_by_1, 
			bet_by_opponent_ : bet_by_2,
			raise_ : raise_amount})

	#p2 goes now
	p2_fold_logit, p2_raise, p2_raise_amount = session.run(
		policy2, 
		feed_dict = {
			x_ : x2
			money_ : p2_money,
			round_num_ : turn,
			bet_by_me_ : bet_by_2, 
			bet_by_opponent_ : bet_by_1,
			raise_ : raise_amount})



"""
lets do an example with two networks

inputs:

x_ : x1,
money_ : p1_money,
round_num_ : turn,
bet_by_me_ : bet_by_1, 
bet_by_opponent_ : bet_by_2,
raise_ : raise_amount})

output:

p1_fold, p1_raise, p1_raise_amount

"""


x_ = tf.placeholder(
	dtype = tf.float32,
	shape = [None, 1], 
	name = "hand"
	)

money_ = tf.placeholder(
	dtype = tf.float32,
	shape = [None, 1], 
	name = "money"
	)

round_num_ = tf.placeholder(
	dtype = tf.int23,
	shape = [None, 1], 
	name = "round_number"
	)

bet_by_me_ = tf.placeholder(
	dtype = tf.float32,
	shape = [None, 1], 
	name = "bet_by_me"
	)

bet_by_opponent_ = tf.placeholder(
	dtype = tf.float32,
	shape = [None, 1], 
	name = "bet_by_opponent"
	)

raise_ = tf.placeholder(
	dtype = tf.float32,
	shape = [None, 1], 
	name = "raise_amount"
	)


input_vector = tf.concat(
	[x_, money_, my_bet_, bet_by_opponent_, raise_, tf.one_hot(round_, 3)]
	)


#network 1
n1_layer_1 = tf.layers.Dense(
	units = 500,
	activation = tf.nn.relu 
	)

n1_layer_2 = tf.layers.Dense(
	units = 3,
	activation = None 
	)


#network 2
n2_layer_1 = tf.layers.Dense(
	units = 500,
	activation = tf.nn.relu 
	)

n2_layer_2 = tf.layers.Dense(
	units = 3,
	activation = None 
	)


#loss placeholder


#split it between two applications

n1_layer_1 = n1_layer_1(input_vector)
G1 = n1_layer_2(n1_layer_1)

n2_layer_1 = n2_layer_1(input_vector)
G2 = n2_layer_2(n2_layer_1)





