import random
import numpy as np
from treys import Card, Evaluator
from tools import CARD_VECTOR_DICT 

def winning_hand(hole1, hole2, community):
	"""returns which hand is winning
	0 -> player1 wins
	1 -> player2 wins
	"""

	str_to_card = lambda s: Card.new(s)

	hole1 = [str_to_card(s) for s in hole1]
	hole2 = [str_to_card(s) for s in hole2]
	community = [str_to_card(s) for s in community]

	evaluator = Evaluator()

	try:
		score1 = evaluator.evaluate(community, hole1)
		score2 = evaluator.evaluate(community, hole2)
	except:
		print('error')
		print(Card.print_pretty_cards(hole1))
		print(Card.print_pretty_cards(hole2))
		print(Card.print_pretty_cards(community))


	if score1>score2:
		return 0
	elif score2>score1:
		return 1
	else:
		assert(score1 == score2)
		return 0.5

class Deck():
	"""make a deck of cards from which to draw
	"""
	def __init__(self):
		self.R = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
		self.S = ['h','s','c','d']
		self.cards = []
		for r in self.R:
			for s in self.S:
				self.cards.append(r+s)
		random.shuffle(self.cards)
		assert(len(self.cards)==52)

	def reshuffle(self):
		self.cards = []
		for r in self.R:
			for s in self.S:
				self.cards.append(r+s)
		random.shuffle(self.cards)

	def draw(self, n):
		#draws n cards. If we run out of cards, we refill cards.
		out = []
		for _ in range(n):
			if self.cards:
				out.append(self.cards.pop())
			else:
				self.reshuffle()
				out.append(self.cards.pop())
		return out

	def pull_cards(self, cards_to_pull):
		"""
		reshuffles deck, 
		then pulls cards from deck
		cards is a list naming the cards to take from the deck
		"""
		self.reshuffle()
		out = []
		for card in cards_to_pull:
			self.cards.remove(card)
			out.append(card)
		return out

def rank_hands(holes, community):
	"""returns which hand is winning
	0 -> player1 wins
	1 -> player2 wins
	"""

	str_to_card = lambda s: Card.new(s)
	processed_holes = []

	for hole in holes:
		processed_holes.append([str_to_card(s) for s in hole])
	
	community = [str_to_card(s) for s in community]
	evaluator = Evaluator()
	scores = []

	try:
		for hole in processed_holes:
			scores.append(evaluator.evaluate(community, hole))
		
	except:
		print('error')
		end


	combined = list(enumerate(scores))

	combined.sort(key = lambda x: x[-1])

	#print(combined)

	order, valuesorder = zip(*combined)

	return order

#print('this', rank_hands([['Ah', 'As'], ['2c', 'Th'], ['2s', '3h']], ['Tc', '4d', '7s', 'Ts', 'Td']))
	


class poker_game():
	"""This class tracks the progress of the game

	"""
	def __init__(self, num_players):
		"""Set up board for num_players players
		"""
		init_money = 10

		self.num_players = num_players

		assert(num_players>1)
		deck = Deck()
		self.holes = [deck.draw(2) for _ in range(num_players)]
		self.community = deck.draw(5)
		self.flop = self.community[:3]
		self.river = self.community[:4]
		self.turn = self.community[:5]

		#sort hands
		self.player_ranks = rank_hands(self.holes, self.community)
		self.still_playing = [True for _ in range(num_players)]

		self.state = 0 #pre-flop, flop, turn, river
		self.current_player = 0 #which player is currently playing

		self.pot = 0
		self.money = [init_money for _ in range(num_players)]

		self.pending_raise = [0, 0] #of the form [raise amount, player who initiated raise]
		self.current_bets = [0 for _ in range(num_players)] #bets this turn by all players
		self.total_bets = [0 for _ in range(num_players)] #bets overall by all players
		self.raise_reps = [0 for _ in range(num_players)] #how many times has this player already raised

		self.player_calls = [False for _ in range(num_players)]

		self.player_last_moves = [[0,0,0] for _ in range(num_players)] #call fold raise [1,0,0], [0,1,0], [0,0,1]

		self.reward_gradients = [None for _ in range(num_players)]

		self.reward_gradient_multipliers = [0 for _ in range(num_players)] #the number of individuals who call a particular raise
		self.pending_reward_gradients = [None for _ in range(num_players)]
		
		self.c_grads = [None for _ in range(num_players)]
		self.f_grads = [None for _ in range(num_players)]
		self.r_grads = [None for _ in range(num_players)]

		self.winner = None

	def add_gradient(self, gradient, target, multiplier = 1):
		"""Takes gradient and adds it to the target
		"""
		#assert((target != None) or (gradient != None))

		if target == None:
			#no gradient added yet
			target = gradient 

		if gradient == None:
			pass
	
		else:
			#lets add to the target
			for i, v in enumerate(gradient):
				target[i] += v*multiplier

		return target


	def vectorize_hands(self, holes, communities):
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

			vectors.append(v)

		return np.array(vectors)

	def get_hand_vector(self, i):
		"""gives the vector for my hand
		"""
		if self.state == 0:
			#pre flop
			return self.vectorize_hands([self.holes[i]], [[]])

		if self.state == 1:
			#flop
			return self.vectorize_hands([self.holes[i]], [self.flop])

		if self.state == 2:
			#turn
			return self.vectorize_hands([self.holes[i]], [self.turn])

		if self.state >= 3:
			#river
			return self.vectorize_hands([self.holes[i]], [self.river])

	def game_state(self, i):
		"""returns the values which define the state of the 
		board for the nth player
		these can be processed and fed into the feed_dict 
	
		including:

		hand vector #a vector of the players current hand 
		self.money #the players total money
		self.bet_by_me #amount of money bet by me
		self.pot_total_ #amount in the pot
		self.raise #raise amount being asked for
		self.round_num_ #0 preflop, 1 flop, 2 turn, 3 river, 4 finished
		self.num_players

		Now for the repeated fields
		these are given for every player but the current one

		fold bool
		call_bool
		last move
		current bets
		total bets
		still have to play
		"""

		output_dict = {
			'player_num' : self.num_players,
			'hand_vector' : self.get_hand_vector(i),
			'money' : self.money[i],
			'bet_by_me' : self.current_bets[i],
			'pot' : self.pot,
			'raise' : self.pending_raise[0] - self.current_bets[i], #note that this is the current raise amount to consider
			'round_num' : self.state,
			'fold_bools' : [[int(b)] for index, b in enumerate(self.still_playing) if index!=i],
			'call_bools' : [[int(b)] for index, b in enumerate(self.player_calls) if index!=i],
			'last_move' : [cfr for index, cfr in enumerate(self.player_last_moves) if index!=i],
			'current_bets' : [[bet] for index, bet in enumerate(self.current_bets) if index!=i],
			'total_bets' : [[bet] for index, bet in enumerate(self.total_bets) if index!=i],
			'not_yet_played' : [[int(b)*int(self.player_calls[index])] for index, b in enumerate(self.still_playing) if index!=i]
		}
		#note not yet played is determined by 
		#still playing and not called

		return output_dict

	def action(self, call_bool, fold_bool, raise_bool, raise_amount, decision_gradient, raise_gradient):
		"""Apply the appropriate next action in the game
		"""
		#check that theres not onely one player
		#print('acting')

		if sum(self.still_playing)==1:
			#print('one left')
			for i, b in enumerate(self.still_playing):
				if b:
					self.winner = i
		
		#check that self.state is not 4, ie the game is not over
		elif self.state != 4:
			#print('not state 4')
			if call_bool:
				self.call_action(decision_gradient)

			if fold_bool:
				self.fold_action(decision_gradient)

			if raise_bool:
				self.raise_action(raise_amount, decision_gradient, raise_gradient)

			self.advance()
			

		else:
			#determine winner
			#allocate funds
			#print('done playing, who won')
			winner = -1
			rank = 10**4
			
			
			for i in range(self.num_players):
				if self.still_playing[i]:
					
					if self.player_ranks[i]<rank:
						winner = i 
						rank = self.player_ranks[i]
			
			self.winner = winner

		#If we have found a winner, give him the pot
		if self.winner != None:
			self.money[self.winner] += self.pot 


		return self.winner

	def call_action(self, decision_gradient):
		"""Call the current hand.
		update state of board

		decision gradient is the gradient of the call probability decision
		"""

		#update state of board
		if (self.pending_raise[0]-self.current_bets[self.current_player])>0:
			#there is currently a pending raise

			#update money, pot
			self.money[self.current_player] -= (self.pending_raise[0]-self.current_bets[self.current_player])
			self.pot += float(self.pending_raise[0]-self.current_bets[self.current_player])
			self.current_bets[self.current_player]= self.pending_raise[0]
			self.total_bets[self.current_player]+=(self.pending_raise[0]-self.current_bets[self.current_player])

			#add to self.reward_gradient_multipliers
			self.reward_gradient_multipliers[self.pending_raise[-1]]+=1

		#add call gradient to call gradients
		self.c_grads[self.current_player] = self.add_gradient(decision_gradient, self.c_grads[self.current_player])

		self.player_calls[self.current_player] = True
		self.player_last_moves[self.current_player] = [1, 0, 0]

	def fold_action(self, decision_gradient):
		"""fold hand
		update state of board

		decision gradient is the gradient of the fold probability decision
		"""

		#update state of game 
		self.still_playing[self.current_player] = False
		self.player_last_moves[self.current_player] = [0, 1, 0]

		#update gradients
		self.f_grads[self.current_player] = self.add_gradient(decision_gradient, self.f_grads[self.current_player])

	def raise_action(self, raise_amount, decision_gradient, raise_gradient):
		"""fold hand
		update state of board

		we treat this as a call, and then a raise

		decision gradient is the gradient of the fold probability decision
		raise gradient is the gradient of the raise amount
		"""

		#here we check if we've raised N=10 times already
		#if so, we only call
		if self.raise_reps[self.current_player]>10:
			self.call_action(decision_gradient)

		else:
			#update state of board

			#first deal with current raise amount
			if self.pending_raise[0]:
				#there is currently a pending raise

				#add to self.reward_gradient_multipliers
				self.reward_gradient_multipliers[self.pending_raise[-1]]+=1

				#add pending reward to reward - multiplied by the appropriate factor
				self.reward_gradients[self.pending_raise[-1]] = self.add_gradient(
					self.pending_reward_gradients[self.pending_raise[-1]], 
					self.reward_gradients[self.pending_raise[-1]], 
					self.reward_gradient_multipliers[self.pending_raise[-1]])

				#clear pending reward gradients
				self.pending_reward_gradients[self.pending_raise[-1]] = None
				self.reward_gradient_multipliers[self.pending_raise[-1]] = 0

			#add decision gradient to raise gradients
			self.r_grads[self.current_player] = self.add_gradient(decision_gradient, self.r_grads[self.current_player])

			#update the reward gradients
			self.pending_reward_gradients[self.current_player] = raise_gradient
			self.reward_gradient_multipliers[self.current_player] = 0

			#update the pending bet
			self.pending_raise[0] += raise_amount
			self.pending_raise[-1] = self.current_player

			#update state of the board
			#money, pot
			self.money[self.current_player]-= raise_amount
			self.pot+=float(raise_amount)
			self.current_bets[self.current_player]+=raise_amount
			self.total_bets[self.current_player]+=raise_amount

			#Now we remove all previous calls, as now players must respond to this action
			for i in range(self.num_players):
				if i!= self.current_player:
					self.player_calls[i] = False
			#We, however, call
			self.player_calls[self.current_player] = True
			self.player_last_moves[self.current_player] = [0, 0, 1]
			self.raise_reps[self.current_player]+=1

	def advance(self):
		"""check the state of the board

		do we need to advance to the next round of betting?

		do we need to do another round of bets?
		"""

		#if no players in front of the current player are still playing, then we check if 
		#betting is finished. If so, move to the next round
		#print('advancing')

		if self.round_finished():
			#print('round done')
			self.current_player = 0
			self.state+=1

			self.pending_raise = [0, 0] #of the form [raise amount, player who initiated raise]
			self.current_bets = [0 for _ in range(self.num_players)] #bets this turn by all players
			self.raise_reps = [0 for _ in range(self.num_players)] #no raises for this round yet (this prevents infinite raises)

			self.player_calls = [False for _ in range(self.num_players)]

		else:
			#rounds not over
			#print('not done')
			self.current_player = self.next_player()
		
	def next_player(self):
		"""Are there players waiting to go?
		"""
		#if no players in front of the current player are still playing, then we check if 
		#betting is finished. 
		for i in range(self.current_player+1, self.num_players):
			if self.still_playing[i] and not self.player_calls[i]:
				#we have players that need to play
				return i

		for i in range(self.current_player):
			if self.still_playing[i] and not self.player_calls[i]:
				#we have players that need to play
				return i

		assert(False)

		return -1

	def round_finished(self):
		"""Returns True if betting is over
		else False
		"""
		"""
		#if no players in front of the current player are still playing, then we check if 
		#betting is finished. If so, move to the next round
		for i in range(self.current_player+1, self.num_players):
			if self.still_playing[i]:
				#we have players that need to play
				return False"""
				
		#for betting to be finished, we need all players to
		#either have called or folded in a given round
		for i in range(self.num_players):
			#if ith player is still playing
			if self.still_playing[i]:
				#if the player doesnt call, return false
				if not self.player_calls[i]:
					return False

		#else return true
		return True 











