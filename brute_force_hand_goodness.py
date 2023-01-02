"""
hand goodness isnt really an interesting problem
I'm more interested in the RL aspect of this

Here i'll try to just make a dictionary of probabilities
"""
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


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

#define the data generating function

def winning_hand(hole1, hole2, community):
	"""returns which hand is winning
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
	return min(((score1, 0), (score2, 1)))[-1]

#should be relatively small
two_card_dictionary = dict()

def equivalence_class_2_cards(hole):
	"""
	This is for accessing the corresponding object in the dictionary
	it maps elements to their equivalence class
	"""
	c1, c2 = hole 

	#order
	ranks = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
	i1 = ranks.index(c1[0])
	i2 = ranks.index(c2[0])

	#we want the order to be such that it goes from least to most valuable,
	#otherwise, swap
	if i1>i2:
		c1, c2 = c2, c1

	#same suit
	if c1[-1] == c2[-1]:
		c1 = c1[0]+'c'
		c2 = c2[0]+'c'

	#different suit
	#clubs, then diamonds
	else:
		c1 = c1[0]+'c'
		c2 = c2[0]+'d'

	return (c1, c2)


