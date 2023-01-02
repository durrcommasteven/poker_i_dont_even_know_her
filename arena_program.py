import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import random
import datetime
import poker_tools
from poker_ai_v0 import *
from time import time


class arena():
	"""This is a class to facilitate the managing of 
	multiple tensorflow graphs and sessions
	"""
	def __init__(self, players):
		"""players is a list of poker_players
		"""
		self.players = players 
		self.sessions = [tf.Session(graph = self.players[i].graph) for i, p in enumerate(self.players)]

	def initialize(self, train_new = False):
		"""tries to load parameters for all players
		 if that fails, then initialize them
		"""
		if train_new:
			for i, p in enumerate(players):
				p.init_global_vars(self.sessions[i])

		else:
			for i, p in enumerate(players):
				try:
					p.load(self.sessions[i])
				except:
					p.init_global_vars(self.sessions[i])

	def close(self):
		"""Close all sessions
		"""
		for i, sess in enumerate(self.sessions):
			self.sessions[i].close()

	def output(self, index, game_state = None):
		"""Returns the output of the indexed player
		"""
		if type(game_state) == type(None):
			#then use test values
			feed_dict = self.players[index].test_dict(len(self.players))

			to_eval = (tf.exp(self.players[index].cfr_log_probs), self.players[index].raise_amount)
			return self.sessions[index].run(to_eval, feed_dict = feed_dict)
		
		feed_dict = self.players[index].make_feed_dict(game_state)
		to_eval = (tf.exp(self.players[index].cfr_log_probs), self.players[index].raise_amount)

		#print(self.sessions[index].run(self.players[index].cfr_log_probs, feed_dict = feed_dict))
		return self.sessions[index].run(to_eval, feed_dict = feed_dict)

	def gradients(self, index, cfr, game_state = None):
		"""returns the gradient given cfr and the player index
		"""
		if type(game_state) == type(None):
			feed_dict = self.players[index].test_dict(len(self.players))
		else:
			feed_dict = self.players[index].make_feed_dict(game_state)

		if cfr == 0:
			decision_grad = self.sessions[index].run(self.players[index].c_derivs, feed_dict = feed_dict)
		elif cfr == 1:
			decision_grad = self.sessions[index].run(self.players[index].f_derivs, feed_dict = feed_dict)
		elif cfr == 2:
			decision_grad = self.sessions[index].run(self.players[index].r_derivs, feed_dict = feed_dict)

		raise_grad = self.sessions[index].run(self.players[index].raise_derivs, feed_dict = feed_dict)

		return decision_grad, raise_grad

#our utility function to start with
def utility(initial_money, final_money):
	"""For now we're just going to make the utility equal
	final_money-initial_money

	ie, essentiall this is the infinite money limit

	return a list giving the utilities
	"""
	return np.array(final_money) - np.array(initial_money)






def play_poker(players, M, training_reps, train_new = False, display= True):
	"""plays reps games of poker
	collects relevant gradients 
	"""

	#create utilities
	g = goodness_predictor()
	v = volatility_predictor()

	#make their sessions
	volatility_sess = tf.Session(graph = v.graph)
	goodness_sess = tf.Session(graph = g.graph)

	v.restore(volatility_sess)
	g.restore(goodness_sess)

	f_v = lambda card_vector: float(v.volatility(volatility_sess, vectors = card_vector))
	f_g = lambda card_vector: float(g.goodness(goodness_sess, vectors = card_vector))

	#create arena
	a = arena(players)
	a.initialize(train_new = train_new)

	utilities = []
	mean_times = []

	for _ in range(training_reps):
		print(_)#int(round(100*_/training_reps))%5==0:
		#print('percent done: ',int(round(100*_/training_reps)))
		#compute gradients, then adjust players
		total_gradients = [None for _ in players]

		mean_utilities = np.array([0. for _ in players])
		t0 = time()

		for _ in range(M):
			#play a poker game, collect the gradients
			game = poker_tools.poker_game(len(players))

			initial_money = game.money[:]

			while game.winner==None:
				#while the game is actually still going
				index = game.current_player
				state = game.game_state(index)

				#set up player's stats
				hand_vec = state['hand_vector']
				players[index].hand_stats = [f_g(hand_vec), f_v(hand_vec)]

				#make decision
				[cfr_probs], [raise_val] = a.output(index, game_state = state)

				#print(cfr_probs)
				#print(game.state)
				#print(raise_val)
				action_index = np.random.choice([0,1,2], p = cfr_probs)
				#print(action_index)

				#print('index: ', index)
				#produce gradients
				decision_grad, raise_grad = a.gradients(index, action_index, game_state = state)
				if type(decision_grad) == type(None):
					print('problem with grad')

				#take action
				action_inputs = [action_index == i for i in range(3)]+[float(raise_val), decision_grad, raise_grad]
				#print()
				#print(action_inputs[:-2])

				game.action(*action_inputs)

				#print("state, ", game.state)

				#print(game.game_state(index))
				#if game.state == 3:
				#	#print(game.community)
				#print(game.still_playing)
				#print(game.holes)

			#produce gradient

			final_money = game.money[:]

			U = utility(initial_money, final_money)
			#print("utility", U, initial_money, final_money)
	

			#add to mean utility
			mean_utilities+= np.array(U)/float(M)

			decision_grads = list(zip(game.c_grads, game.f_grads, game.r_grads)) 

			for i, grad_type in enumerate(decision_grads):
				for D in grad_type:
					total_gradients[i] = game.add_gradient(D, total_gradients[i], U[i]/float(M))

			for i, D in enumerate(game.reward_gradients):
				total_gradients[i] = game.add_gradient(D, total_gradients[i], 1./float(M))
		mean_times.append((time()-t0)/M)
		print('time per game: ', ((time()-t0)/M))
		np.save('mean_times.npy', mean_times)

		utilities.append(mean_utilities)
		np.save('utilities.npy', utilities)
		#adjust weights
		for i, p in enumerate(players):
			if i==0:
				players[i].train_step(total_gradients[i], a.sessions[i], dx = 0.01)

			#a.sessions[i].run(players[i].train_op, feed_dict = {players[i].grad_placeholder : grads_and_vars})
			#players[i].train_op(total_gradients[i], a.sessions[i], players[i].graph, dx = 0.01)

	for i, p in enumerate(players):
		p.save(a.sessions[i])
	
	a.close()

	if display:
		xs = zip(*utilities)
		for x in xs:
			plt.plot(x)
			plt.show()




		



		



players = [poker_player_v0('p1'), poker_player_v0('p2'), poker_player_v0('p3')]#, poker_player_v0('p4')]


play_poker(players, M = 10, training_reps = 240, train_new = False)



#sessions = [tf.Session(graph = players[0].graph), tf.Session(graph = players[1].graph)]

#players[0].init_global_vars(sessions[0])

#x1 = sessions[0].run(players[0].nn_output, feed_dict = players[0].test_dict(2))
#print(x1)

"""
a = arena(players)
a.initialize()

print(a.output(0, game_state = None))
print(a.output(1, game_state = None))
print(a.output(0, game_state = None))"""
"""
game = poker_tools.poker_game(2)
players = [poker_player_v0('p1'), poker_player_v0('p2'),]
graphs = [tf.Graph(), tf.Graph()]
sessions = [tf.Session(graph = graphs[0]), tf.Session(graph = graphs[1])]

players[0].init_global_vars(sessions[0])
players[1].init_global_vars(sessions[1])

x1 = sessions[0].run(players[0].nn_output, feed_dict = players[0].test_dict(2))
print(x1)
x2 = sessions[1].run(players[1].nn_output, feed_dict = players[1].test_dict(2))
print(x2)

"""
"""
game.game_state(1)
for i in range(200):
	if game.winner:
		print(game.winner)
	else:
		states = [game.game_state(j) for j in range(10)]
		print([states[j]['money'] for j in range(10)])
		print(game.state)
		game.action(*random_move(np.random.rand()))
"""
