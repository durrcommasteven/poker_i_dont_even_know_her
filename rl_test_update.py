import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import random
import datetime
"""
Here I'm going to make a simple, multi-step task and see if i can train it with tensorflow
"""

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


def high_hand(session, vocal = False):
	"""
	in this game, each player gets a number in [0, 1]
	the players then undergo TURNS rounds of betting 
	The reward is the profit of each player. 

	policy1 and policy2 are networks 

	the outputs are the gradients obtained for both networks

	inputs to the policy are:

	x_ : x1,
	money_ : p1_money,
	round_num_ : turn,
	bet_by_me_ : bet_by_1, 
	bet_by_opponent_ : bet_by_2,
	raise_ : raise_amount

	outputs:
	p1_fold_logit, p1_raise_logit, p1_raise_amount

	policy1 always starts first

	"""
	TURNS = 4
	INIT_POT = 1
	INIT_MONEY = 0

	pot = INIT_POT #start out with something in the pot

	p1_money = INIT_MONEY
	p2_money = INIT_MONEY

	bet_by_1 = 0
	bet_by_2 = 0

	raise_amount = 0.

	#initial hands
	x1 = np.random.rand()
	x2 = np.random.rand()


	p1_c_grads, p1_f_grads, p1_r_grads = [], [], []

	p2_c_grads, p2_f_grads, p2_r_grads = [], [], []

	p1_actions, p2_actions = [], []

	p1_return_grads = []
	p1_pending_raise_grad = None

	p2_return_grads = []
	p2_pending_raise_grad = None

	p1_fold = False
	p2_fold = False

	for turn in range(TURNS):
		#print(bet_by_2)

		#p1 starts
		p1_cfr_log_probs_eval, p1_raise_amount_eval, p1_c_derivs_eval, p1_f_derivs_eval, p1_r_derivs_eval, p1_raise_derivs_eval = session.run(
			(
				p1_cfr_log_probs, 
				p1_raise_amount, 
				p1_c_derivs, p1_f_derivs, p1_r_derivs,
				p1_raise_derivs,
			), 
			feed_dict = {
				x_ : x1,
				money_ : p1_money,
				round_num_ : turn,
				bet_by_me_ :bet_by_1, 
				bet_by_opponent_ : bet_by_2,
				raise_ : raise_amount})
		#print("grads:")
		#print(p1_c_derivs_eval, p1_f_derivs_eval, p1_r_derivs_eval)

		p1_cfr_log_probs_eval = p1_cfr_log_probs_eval.squeeze() 
		p1_raise_amount_eval = p1_raise_amount_eval.squeeze()

		#calculate actions
		#print(np.exp(p1_cfr_log_probs_eval))

		p1_action = np.random.choice(['c', 'f', 'r'], p = np.exp(p1_cfr_log_probs_eval))
		#print(p1_action)
		p1_actions.append(p1_action)

		if p1_action == 'c':
			#player 1 calls

			#first update derivatives
			if raise_amount:
				#add raised mount to p2's return grads
				p2_return_grads.append(p2_pending_raise_grad)
			else:
				#call without raising anything 
				pass

			#update call grads
			p1_c_grads.append(p1_c_derivs_eval)

			#next update game state
			p1_money -= raise_amount
			pot += raise_amount
			bet_by_1 += raise_amount
			raise_amount = 0

		if p1_action == 'r':
			#player 1 raises

			if raise_amount:
				#then we are raising, and the oponent already raised

				#first we essentially go through the process of calling
				#update state for this initial raise amount
				p1_money -= raise_amount
				pot+=raise_amount
				bet_by_1+=raise_amount

				#add raised amount to p2's return grads
				p2_return_grads.append(p2_pending_raise_grad)

			#add p1 raise deriv to p1 pending raise
			p1_pending_raise_grad = p1_raise_derivs_eval

			#update raise grads
			p1_r_grads.append(p1_r_derivs_eval)

			#update game state with new raise amount
			raise_amount = p1_raise_amount_eval
			p1_money -= raise_amount
			pot += raise_amount
			bet_by_1 += raise_amount

		if p1_action == 'f':
			"""
			Given a fold, the match ends, and we calculate the winner
			"""
			p1_f_grads.append(p1_f_derivs_eval)
			p1_fold = True
			break

		#Now player 2 responds
		p2_cfr_log_probs_eval, p2_raise_amount_eval, p2_c_derivs_eval, p2_f_derivs_eval, p2_r_derivs_eval, p2_raise_derivs_eval = session.run(
			(
				p2_cfr_log_probs, 
				p2_raise_amount, 
				p2_c_derivs, p2_f_derivs, p2_r_derivs,
				p2_raise_derivs,
			), 
			feed_dict = {
				x_ : x2,
				money_ : p2_money,
				round_num_ : turn,
				bet_by_me_ : bet_by_2, 
				bet_by_opponent_ : bet_by_1,
				raise_ : raise_amount})

		p2_cfr_log_probs_eval = p2_cfr_log_probs_eval.squeeze() 
		p2_raise_amount_eval = p2_raise_amount_eval.squeeze()
		#calculate actions

		p2_action = np.random.choice(['c', 'f', 'r'], p = np.exp(p2_cfr_log_probs_eval))

		p2_actions.append(p2_action)

		if p2_action == 'c':
			#player 2 calls

			#first update derivatives
			if raise_amount:
				#add raised mount to p1's return grads
				p1_return_grads.append(p1_pending_raise_grad)
			else:
				#call without raising anything 
				pass

			#update call grads
			p2_c_grads.append(p2_c_derivs_eval)

			#next update game state
			p2_money -= raise_amount
			pot+=raise_amount
			bet_by_2 += raise_amount
			raise_amount = 0

		if p2_action == 'r':
			#player 2 raises

			if raise_amount:
				#then we are raising, and the oponent already raised

				#first we essentially go through the process of calling
				#update state for this initial raise amount
				p2_money -= raise_amount
				pot+=raise_amount
				bet_by_2+=raise_amount

				#add raised amount to p1's return grads
				p1_return_grads.append(p1_pending_raise_grad)

			#add p2 raise deriv to p2 pending raise
			p2_pending_raise_grad = p2_raise_derivs_eval

			#update raise grads
			p2_r_grads.append(p2_r_derivs_eval)

			#update game state with new raise amount
			raise_amount = p2_raise_amount_eval
			p2_money -= raise_amount
			pot += raise_amount
			bet_by_2 += raise_amount

		if p2_action == 'f':
			"""
			Given a fold, the match ends, and we calculate the winner
			"""
			p2_f_grads.append(p2_f_derivs_eval)
			p2_fold = True
			break

	"""
	GAME OVER
	calculate gradients
	"""

	#who gets the pot:
	#if one player folded and the other didnt, the other gets the pot
	#if neither players folded, the higher hand gets the pot

	winner = None #an integer for the winner

	#did somebody fold
	if p1_fold or p2_fold:
		if not p1_fold:
			winner = 0
		else:
			winner = 1
	else:
		#winner has higher hand
		winner = max((x1, 0), (x2, 1))[1]

	if winner == 0:
		#player 1 wins
		p1_reward = p1_money+pot - INIT_MONEY
		#print('p1 reward', p1_reward)

		p1_policy_grads = sum_gradients(p1_c_grads+p1_f_grads+p1_r_grads)*p1_reward

		p1_reward_grads = sum_gradients(p1_return_grads)

		p2_reward = p2_money - INIT_MONEY

		p2_policy_grads = sum_gradients(p2_c_grads+p2_f_grads+p2_r_grads)*p2_reward

		p2_reward_grads = -1*sum_gradients(p2_return_grads)

	if winner == 1:
		#player 1 wins
		p2_reward = p2_money + pot - INIT_MONEY

		p2_policy_grads = sum_gradients(p2_c_grads+p2_f_grads+p2_r_grads)*p2_reward

		p2_reward_grads = sum_gradients(p2_return_grads)

		p1_reward = p1_money - INIT_MONEY

		p1_policy_grads = sum_gradients(p1_c_grads+p1_f_grads+p1_r_grads)*p1_reward

		p1_reward_grads = -1*sum_gradients(p1_return_grads)

	#give gradients
	if vocal:
		print(p1_actions)
		print(p2_actions)
		print(x1, p1_reward)
		print(x2, p2_reward)

	p1_grads = p1_policy_grads + p1_reward_grads
	p2_grads = p2_policy_grads + p2_policy_grads

	return p1_grads, p2_grads, p1_reward, p2_reward



"""

def raise_by_1(policy1, policy2, session):
	

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
		p1_fold_logit, p1_raise, p1_raise_amount, 
		feed_dict = {
			x_ : x1,
			money_ : p1_money,
			round_num_ : turn,
			bet_by_me_ : bet_by_1, 
			bet_by_opponent_ : bet_by_2,
			raise_ : raise_amount})

	#p2 goes now
	p2_fold_logit, p2_raise, p2_raise_amount = session.run(
		p2_fold_logit, p2_raise, p2_raise_amount, 
		feed_dict = {
			x_ : x2
			money_ : p2_money,
			round_num_ : turn,
			bet_by_me_ : bet_by_2, 
			bet_by_opponent_ : bet_by_1,
			raise_ : raise_amount})

	f1 = tf.log()
"""








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
	shape = [], 
	name = "hand"
	)

money_ = tf.placeholder(
	dtype = tf.float32,
	shape = [], 
	name = "money"
	)

round_num_ = tf.placeholder(
	dtype = tf.int32,
	shape = [], 
	name = "round_number"
	)

bet_by_me_ = tf.placeholder(
	dtype = tf.float32,
	shape = [], 
	name = "bet_by_me"
	)

bet_by_opponent_ = tf.placeholder(
	dtype = tf.float32,
	shape = [], 
	name = "bet_by_opponent"
	)

raise_ = tf.placeholder(
	dtype = tf.float32,
	shape = [], 
	name = "raise_amount"
	)

input_vector = tf.reshape(
	tf.concat(
		[[
			x_, money_, 
			bet_by_me_, 
			bet_by_opponent_], 
			tf.reshape(tf.one_hot(round_num_, 4), [4]),
		], 0
	), [-1, 8])

input_vector.shape


#network 1
n1_layer_1 = tf.layers.Dense(
	units = 500,
	activation = tf.nn.relu,
	name = 'network_1_dense_1'
	)

n1_layer_2 = tf.layers.Dense(
	units = 4,
	activation = None,
	name = 'network_1_dense_2' 
	)

n1_layer_1 = n1_layer_1(input_vector)
n1_output = n1_layer_2(n1_layer_1)


p1_cfr_logit, p1_raise_amount_unscaled = tf.split(n1_output, [3,1], 1)
p1_cfr_log_probs = tf.nn.log_softmax(p1_cfr_logit)

#log probs
p1_c_log_probs, p1_f_log_probs, p1_r_log_probs = tf.split(p1_cfr_log_probs, [1,1,1], 1)
#raise amount
p1_raise_amount = tf.sigmoid(p1_raise_amount_unscaled) #just for now we limit bets

#derivatives
vars_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network_1')

p1_c_derivs = tf.gradients(p1_c_log_probs, vars_1)
p1_f_derivs = tf.gradients(p1_f_log_probs, vars_1)
p1_r_derivs = tf.gradients(p1_r_log_probs, vars_1)

p1_raise_derivs = tf.gradients(p1_raise_amount, vars_1)

#network 2

#network 2
n2_layer_1 = tf.layers.Dense(
	units = 500,
	activation = tf.nn.relu,
	name = 'network_2_dense_1')

n2_layer_2 = tf.layers.Dense(
	units = 4,
	activation = None,
	name = 'network_2_dense_2'
	)

n2_layer_1 = n2_layer_1(input_vector)
n2_output = n2_layer_2(n2_layer_1)



p2_cfr_logit, p2_raise_amount_unscaled = tf.split(n2_output, [3,1], 1)
p2_cfr_log_probs = tf.nn.log_softmax(p2_cfr_logit)

#log probs
p2_c_log_probs, p2_f_log_probs, p2_r_log_probs = tf.split(p2_cfr_log_probs, [1,1,1], 1)
#raise amount
p2_raise_amount = tf.sigmoid(p2_raise_amount_unscaled) #just for now we limit bets

#derivatives
vars_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network_2')


p2_c_derivs = tf.gradients(p2_c_log_probs, vars_2)
p2_f_derivs = tf.gradients(p2_f_log_probs, vars_2)
p2_r_derivs = tf.gradients(p2_r_log_probs, vars_2)

p2_raise_derivs = tf.gradients(p2_raise_amount, vars_2)


saver = tf.train.Saver()
load = False

p1_rewards = []
p2_rewards = []

SAMPLE_SIZE = 300

training_reps = 10**5

with tf.Session() as sess:
	if load:
		saver.restore(sess, "/tmp_rl_test/model_most_recent.ckpt")
	else:
		sess.run(tf.global_variables_initializer())

	for training_rep in range(training_reps):
		#print(training_rep)
		if training_rep % (training_reps//100) == 0:
			print(" ")
			print(datetime.datetime.now())
			print("training progress: "+str(training_rep/training_reps))

		temp_p1_mean_reward = 0
		temp_p2_mean_reward = 0

		p1_mean_grad = None
		p2_mean_grad = None

		for rep in range(SAMPLE_SIZE):
			p1_grad, p2_grad, p1_reward, p2_reward = high_hand(sess, vocal = False)

			temp_p1_mean_reward += p1_reward/SAMPLE_SIZE
			temp_p2_mean_reward += p2_reward/SAMPLE_SIZE

			if type(p1_mean_grad) == type(None):
				p1_mean_grad = p1_grad/SAMPLE_SIZE
			else:
				p1_mean_grad += p1_grad/SAMPLE_SIZE

			if type(p2_mean_grad) == type(None):
				p2_mean_grad = p2_grad/SAMPLE_SIZE
			else:
				p2_mean_grad += p2_grad/SAMPLE_SIZE

		#print(temp_p1_mean_reward, temp_p2_mean_reward)
		p1_rewards.append(temp_p1_mean_reward)
		p2_rewards.append(temp_p2_mean_reward)

		p1_grads_and_vars = list(zip(-1*p1_mean_grad, vars_1))
		sess.run(tf.train.GradientDescentOptimizer(0.01).apply_gradients(p1_grads_and_vars))

		p2_grads_and_vars = list(zip(-1*p2_mean_grad, vars_2))
		sess.run(tf.train.GradientDescentOptimizer(0.01).apply_gradients(p2_grads_and_vars))

		np.save('p1_rewards.npy', p1_rewards)
		np.save('p2_rewards.npy', p2_rewards)

plt.plot(p1_rewards, c = 'b')
plt.plot(p2_rewards, c = 'r')
plt.show()


	

