"""
Here I'm going to train a neural network to predict the goodness of a 
hand. This means

goodness(hole_cards, community_cards) = G

this is such that 

given two players with different hole cards, 
the probability that player 1 has a better hand than 
player 2 is 

sigmoid(G1-G2)


So what i have to do is 

-make a function which produces data:
this involves 
playing through whole runs 

ie producing data of the form 
[card11, card12], []
[card21, card22], []

[card11, card12], [c1, c2, c3]
[card21, card22], [c1, c2, c3]

[card11, card12], [c1, c2, c3, c4]
[card21, card22], [c1, c2, c3, c4]

[card11, card22], [c1, c2, c3, c4, c5]
[card21, card22], [c1, c2, c3, c4, c5]

along with an indicator of who won l = 0 or 1

the neural network outputs a goodness G

we apply it to two sets of hole cards and obtain G1 and G2
then we try to minimize

Loss = tf.nn.sigmoid_cross_entropy_with_logits(
	logits = G1-G2,
	labels = l
	)

"""
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import random
from treys import Card, Evaluator
from tools import CARD_VECTOR_DICT 
import datetime


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


def vectorize_data(hole1, hole2, community, normalize = False):
	#takes (hole1, hole2, community)
	#outputs corresponding 53 dim vectors

	v1, v2 = np.zeros(105), np.zeros(105)

	for c in community:
		v1[52+CARD_VECTOR_DICT[c]]+=1
		v2[52+CARD_VECTOR_DICT[c]]+=1

	for _ in range(5-len(community)):
		v1[104]+=1
		v2[104]+=1

	for c in hole1:
		v1[CARD_VECTOR_DICT[c]]+=1

	for c in hole2:
		v2[CARD_VECTOR_DICT[c]]+=1

	if normalize:
		v1 -= np.mean(v1)
		v2 -= np.mean(v2)

		v1 /= np.std(v1)
		v2 /= np.std(v2)

	return v1, v2
	

def data_label(n, vocal = False):
	"""
	give n data label pairs

	whether to train on 
	the flop<=>3
	the turn<=>4
	the river<=>5

	is given by flop_turn_river
	and can be changed 
	"""

	flop_turn_river = lambda : random.choice([0,0,0])

	data1 = [] # vectors of length 105
	data2 = [] # vectors of length 105
	labels = [] # int 1 or 0

	deck = Deck()

	for hand in range(n):
		deck.reshuffle()

		hole1 = deck.draw(2)
		hole2 = deck.draw(2)
		community = deck.draw(5)
		unobscured_community = community[:flop_turn_river()]

		if vocal:
			print("")
			print('hole 1, hole 2, community:')
			print(hole1, hole2, unobscured_community)

		label = winning_hand(hole1, hole2, community)

		v1, v2 = vectorize_data(hole1, hole2, unobscured_community)

		data1.append(v1)
		data2.append(v2)
		labels.append([label])

	data1, data2, labels = np.array(data1), np.array(data2), np.array(labels)

	return data1, data2, labels 


def prediction_test(hole_cards, community_cards, n, vocal = False):
	"""
	here we generate n games of poker given 
	hole cards
	community cards

	we then output the percent of the times that the cards win
	"""

	flop_turn_river = lambda : random.choice([0,0,0])

	player_1_wins = [] #1 if player 1 wins, 0 else
	
	deck = Deck()

	for hand in range(n):
		#first pull cards
		pulled_cards = deck.pull_cards(hole_cards+community_cards)
		hole1 = pulled_cards[:2]

		#then draw other pair of hole cards
		hole2 = deck.draw(2)

		#next finish pulling community cards
		community = pulled_cards[2:] + deck.draw(5 - len(community_cards))

		unobscured_community = community[:flop_turn_river()]

		label = winning_hand(hole1, hole2, community)

		if vocal:
			print("")
			print('hole 1, hole 2, community:')
			print(hole1, hole2, unobscured_community)
			print('player 1 wins: ', bool(label))

		
		#print(label)

		player_1_wins.append(label)

	return np.mean(player_1_wins)

"""
convergent_value = []
reps = [5**5]#list(map(lambda x: int(round(x)), 10**np.linspace(1, 5, 10)))
print(reps)

for r in reps:
	print(r)
	out = prediction_test(hole_cards = ['Ah', 'Jc'], community_cards = ['3c', '5s', 'Td'], n = r, vocal = False)
	convergent_value.append(out)
	print(out)

plt.scatter(reps, convergent_value)
plt.show()

"""
#here we define the model

x1_ = tf.placeholder(
	dtype = tf.float32,
	shape = [None, 105], 
	name = "intputs"
	)

x2_ = tf.placeholder(
	dtype = tf.float32,
	shape = [None, 105], 
	name = "intputs"
	)

y_ = tf.placeholder(
	dtype = tf.float32,
	shape = [None, 1], 
	name = "labels"
	)

#Make network

layer_1 = tf.layers.Dense(
	units = 1024,
	activation = tf.nn.relu 
	)

hidden_1 = tf.layers.Dense(
	units = 1024,
	activation = tf.nn.relu 
	)

layer_2 = tf.layers.Dense(
	units = 1,
	activation = None 
	)

#split it between two applications

n1_layer_1 = layer_1(x1_)
n1_hidden_1 = hidden_1(n1_layer_1)
G1 = layer_2(n1_layer_1)

n2_layer_1 = layer_1(x2_)
n2_hidden_1 = hidden_1(n2_layer_1)
G2 = layer_2(n2_layer_1)

#Make loss functions
prediction_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
	logits = G1-G2,
	labels = y_
	))

mean_loss = 0.5*(tf.reduce_mean(G1)**2 + tf.reduce_mean(G2)**2)


loss = prediction_loss+mean_loss
#make train step
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

pred_losses = []
mean_losses = []

reps = 10**5
batch_size = 15
check_every = 15

saver = tf.train.Saver()
load = True


extract_weights = True

if extract_weights:
	assert(load)
	with tf.Session() as sess:
		saver.restore(sess, "/tmp_goodness_hidden_1/model_most_recent.ckpt")

		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		names = ['kernel_01.npy', 'bias_01.npy','kernel_11.npy', 'bias_11.npy', 'kernel_21.npy', 'bias_21.npy']

		for i, v in enumerate(variables):
			if i ==0:
				print(sess.run(v)[0])
			new_array = sess.run(v)
			name = 'hand_goodness_trained_weights/'+names[i]
			print(name)
			np.save(name, new_array)

	#end
	#i dont want this to keep going



eval_on_card = True

if eval_on_card:
	with tf.Session() as sess:
		if load:
			saver.restore(sess, "/tmp_goodness_hidden_1/model_most_recent.ckpt")
		else:
			sess.run(tf.global_variables_initializer())

		R = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
		S = ['h','s','c','d']

		R = ['5','A', 'T']
		S = ['h']

		cards = []
		for r in R:
			for s in S:
				cards.append(r+s)

		holes = []
		for c1 in cards:
			for c2 in cards:
				if c1 == c2 or sorted((c1, c2)) in holes:
					continue
				else:
					holes.append(sorted((c1, c2)))

		if len(holes)%2 !=0:
			holes.append(holes[-1])



		best_holes = []



		for i in range(len(holes)//2):
			c1, c2 = holes[2*i]
			c3, c4 = holes[2*i +1]
			v1, v2 = vectorize_data([c1, c2], [c3, c4], [], normalize = False)#['3c', '5s', 'Td'], normalize = False)
			g1, g2 = sess.run((G1, G2), feed_dict = {x1_ : [v1], x2_ : [v2]})

			h1 = (g1.flatten(), [c1, c2])
			h2 = (g2.flatten(), [c3, c4])
			
			

			best_holes.append(h1)
			best_holes.append(h1)


		v1, v2 = vectorize_data(['Ah', 'Jc'], ['Ts', 'Js'], [], normalize = False)#['3c', '5s', 'Td'], normalize = False)
		g1, g2 = sess.run((G1, G2), feed_dict = {x1_ : [v1], x2_ : [v2]})

		#print(g1, g2)

best_holes = sorted(best_holes, reverse = False)
print(best_holes)
#print(best_holes[:-100])

bins = 100

scores, _ = zip(*best_holes)

min_s, max_s = min(scores), max(scores)

hist = [0 for _ in range(bins)]


dx = (max_s-min_s)/bins
for s in scores:
	i = int((s-min_s)//dx)
	try:
		hist[i]+=1
	except:
		hist[-1]+=1

xs = [i*dx + min_s for i in range(bins)]
#print(xs)
plt.plot(xs, hist)
plt.show()
asd






with tf.Session() as sess:
	if load:
		saver.restore(sess, "/tmp_goodness_hidden_1/model_most_recent.ckpt")
	else:
		sess.run(tf.global_variables_initializer())
    
	for r in range(reps):

		if r % (reps//100) == 0:
			print(" ")
			print(datetime.datetime.now())
			print("training progress: "+str(r/reps))

		d1, d2, l = data_label(batch_size, vocal = False)
		sess.run(train_step, feed_dict = {x1_ : d1, x2_ : d2, y_ : l})
		#g1, g2 = sess.run((G1, G2), feed_dict = {x1_ : d1, x2_ : d2, y_ : l})
		#print(g1, g2, 1/(1+np.exp(-(g1-g2))))
		
		#print(sess.run(
		#		(G1, G2), feed_dict = {x1_ : d1, x2_ : d2}))
		if r%check_every==0:
			current_mean_loss = sess.run(
				mean_loss, feed_dict = {x1_ : d1, x2_ : d2})

			current_prediction_loss = sess.run(
				prediction_loss, feed_dict = {x1_ : d1, x2_ : d2, y_ : l})

			pred_losses.append(current_prediction_loss)
			mean_losses.append(current_mean_loss)

			save_path = saver.save(sess, "/tmp_goodness_hidden_1/model_rep_"+str(r)+".ckpt")
			save_path = saver.save(sess, "/tmp_goodness_hidden_1/model_most_recent.ckpt")
			#print(save_path)


plt.plot(pred_losses)
plt.show()

plt.plot(mean_losses)
plt.show()

np.save("prediction_losses.npy", np.array(pred_losses))
np.save("mean_losses.npy", np.array(mean_losses))
