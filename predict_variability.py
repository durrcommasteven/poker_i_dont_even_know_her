"""
Here we assume that we have access to a function which outputs the 'goodness' of a given
hole-community pairing

We then use this to predict how volitile the game is possible to be

Assumptions:
Im going to assume that the goodness throughout turns approximates a random walk

this is based on the idea that all available information is already gleaned by 
the goodness neural network. The remaining changes are random ~ random walk.

We'll try to obtain a given hand's diffusion constant, D

remember 

<(x_{t} - x_{0})^2> = 2 D t

let n be the number of cards in community. Lets use the following

<(x_{n} - x_{0})^2> = 2 D n

It seems like this would make more sense 

---------------------
CHECK THIS
i.e.

is <(x_{n} - x_{0})^2> = 2 D n

This would make sense, treating adding another card as a time step of 'goodness'
---------------------

Assuming this is true, we'll try to determine D for a given hand of cards
"""
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from goodness_function import goodness
from tools import CARD_VECTOR_DICT 
import random
import datetime
import glob
from time import time

k0 = np.load('hand_goodness_trained_weights/kernel_0.npy')
b0 = np.load('hand_goodness_trained_weights/bias_0.npy')
k1 = np.load('hand_goodness_trained_weights/kernel_1.npy')
b1 = np.load('hand_goodness_trained_weights/bias_1.npy')
k2 = np.load('hand_goodness_trained_weights/kernel_2.npy')
b2 = np.load('hand_goodness_trained_weights/bias_2.npy')

print(k0.shape)
print(b0.shape)
print(k1.shape)
print(b1.shape)
print(k2.shape)
print(b2.shape)


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


def vectorize_hands(holes, communities, normalize = False):
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

def relu(array):
	return array*(array>0)

def goodness(holes, communities):
	v = vectorize_hands(holes, communities)

	#l1 = relu(np.matmul(v, k0)+b0)
	l1 = relu(np.matmul(k0.T, v.T)+b0[:, None])

	#l2 = relu(np.matmul(l1, k1)+b1)
	l2 = relu(np.matmul(k1.T, l1)+b1[:, None])

	return np.matmul(k2.T, l2)+b2[:, None]#np.matmul(l2, k2)+b2


"""

x1_ = tf.placeholder(
	dtype = tf.float32,
	shape = [None, 105], 
	name = "intputs"
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

n1_layer_1 = layer_1(x1_)
n1_hidden_1 = hidden_1(n1_layer_1)
G1 = layer_2(n1_layer_1)

saver = tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess, "/updated_tmp_goodness_hidden_1/model_most_recent.ckpt")

	g1 = sess.run(G1, feed_dict = {x1_ : vectorize_hands([['4h', '5c']], [[]])})
	print(g1)


def goodness(vectors = None, holes = None, communities = None, ):
	if type(vectors) != type(None):
		with tf.Session() as sess:
			saver.restore(sess, "/updated_tmp_goodness_hidden_1/model_most_recent.ckpt")

			g1 = sess.run(-G1, feed_dict = {x1_ : vectors})
			return g1
	else:		
		with tf.Session() as sess:
			saver.restore(sess, "/updated_tmp_goodness_hidden_1/model_most_recent.ckpt")

			g1 = sess.run(-G1, feed_dict = {x1_ : vectorize_hands(holes, communities)})
			return g1

"""

R = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
S = ['h','s','c','d']



cs = []
for r in R:
	for s in S:
		cs.append(r+s)

holes = []
for i in range(len(cs)):
	for j in range(len(cs)):
		if i!=j:
			holes.append([cs[i], cs[j]])

gs = goodness(holes, [[] for _ in range(len(holes))])

scores = list(zip(gs, holes))
scores.sort()

print(scores)
ddfg"""


#now I want to determine D
"""do this by drawing hole and community. 
Then determine the goodness over time to get data label pairs
"""


def goodness_data(N, set_n1 = None, init_hole = None, init_community = None):
	"""Returns data of the form

	(g(n2)-g(n1))**2, n2-n1
	

	first it produces data like 
	hand1, hand2, n2-n1

	then it processes it into the final result
	"""

	

	Ns = [0, 3, 4, 5]

	data_1 = []
	data_2 = []
	n_differences = []

	while len(n_differences)<N:

		cards = Deck().draw(7)
		hole = cards[:2]
		community = cards[2:]

		if type(set_n1) == int:
			i = Ns.index(set_n1)
			for n2 in Ns[i+1:]:
				#print(set_n1, n2)
				temp_1 = (hole, community[:set_n1])
				temp_2 = (hole, community[:n2])
				#print(temp_1, temp_2, n2-set_n1)
				n = n2-set_n1

				data_1.append(temp_1)
				data_2.append(temp_2)
				n_differences.append(n)
				#print(n_differences.__len__())

		else:
			for i, n2 in enumerate(Ns):
				for j in range(i):

					temp_1 = (hole, community[:Ns[j]])
					#print(temp_1)
					temp_2 = (hole, community[:n2])
					#print(temp_2)
					n = n2-Ns[j]

					data_1.append(temp_1)
					data_2.append(temp_2)
					n_differences.append(n)

	#process
	g1 = goodness(vectors = vectorize_hands(*zip(*data_1)))
	g2 = goodness(vectors = vectorize_hands(*zip(*data_2)))

	return (g2-g1)**2, n_differences

"""
Alright now lets see if the goodness prediciton behaves as 
brownian motion

i.e.

is <(g_{n} - g_{0})^2> = 2 D n
"""

gdict = dict()

data = zip(*goodness_data(10**5, set_n1=0))

for v, n in data:
	if n in gdict:
		gdict[n][0]+=v
		gdict[n][-1]+=1
	else:
		gdict[n] = [v, 1]

"""
x = [float(i) for i in gdict.keys()]
y = gdict.values()
y = [v[0]/v[1] for v in y]

D= 0.5*(np.mean(np.array(y)/np.array(x)))
print(D)

plt.scatter(x, y)
plt.plot(np.arange(6), 2*D*np.arange(6))
plt.show()
"""

def make_goodness_data_labels(N, name, save = True):
	"""makes data of the form

	vector, (g(n1+1)-g(n1))**2
	

	first it produces data like 
	hand1, hand2

	then it processes it into the final result
	saves it to 

	Note if n1 == 5 then variability is 0
	I'll code that in by hand
	"""

	

	Ns = [0, 3, 4, 5]

	data_1 = []
	data_2 = []
	count = 0

	while count<N:

		cards = Deck().draw(7)
		hole = cards[:2]
		community = cards[2:]

		for i in range(3):
			if count>=N:
				break

			n1, n2 = Ns[i], Ns[i+1]

			#print(n1, n2)

			temp_1 = (hole, community[:n1])
			temp_2 = (hole, community[:n2])
			#print(temp_1, temp_2, n2-n1)

			data_1.append(temp_1)
			data_2.append(temp_2)
			count+=1

	#process

	vectors = vectorize_hands(*zip(*data_1))

	g1 = goodness(vectors = vectorize_hands(*zip(*data_1)))
	g2 = goodness(vectors = vectorize_hands(*zip(*data_2)))

	if save:
		np.save('hand_volatility_data/vectors_'+name+'.npy', np.array(vectors))
		np.save('hand_volatility_data/volatility_'+name+'.npy', 0.5*((g2-g1)**2))

	return np.array(vectors), 0.5*((g2-g1)**2)

make_data = False

if make_data:
	#make 1000 files, each with 1000
	file_num = 537
	N = 1000

	for i in range(file_num):
		make_goodness_data_labels(N, str(i))




def goodness_data_labels(N, custom_index = None):
	"""Retrieves data from memory and serves it
	maybe faster than producing it from scratch

	max index saved for testing 
	"""
	max_index = 536

	if custom_index==None:
		index = np.random.randint(0, max_index)
	else:
		index = custom_index

	vecdata = np.load('hand_volatility_data/vectors_'+str(index)+'.npy')
	voldata = np.load('hand_volatility_data/volatility_'+str(index)+'.npy')

	samples = np.random.randint(0, voldata.shape[0], N)

	return np.take(vecdata, samples, axis = 0), np.take(voldata, samples, axis = 0)




"""
To do at home
maybe its not perfectly linear, but it appears that 
there is an underlying slope for the uncertainty 

whaty does the intercept mean

maybe I can predict the slope

what does it mean for different ns to have different intercepts at different turns
"""



"""
Train on volitility
"""


card_vector_ = tf.placeholder(
	dtype = tf.float32,
	shape = [None, 105], 
	name = "intputs"
	)

volatility_ = tf.placeholder(
	dtype = tf.float32,
	shape = [None, 1], 
	name = "labels"
	)

#Make network

vol_layer_1 = tf.layers.Dense(
	units = 1024,
	activation = tf.nn.relu 
	)

vol_hidden_1 = tf.layers.Dense(
	units = 1024,
	activation = tf.nn.relu 
	)

vol_layer_2 = tf.layers.Dense(
	units = 1,
	activation = None 
	)

vol_net_layer_1 = layer_1(card_vector_)
vol_net_hidden_1 = hidden_1(vol_net_layer_1)
volatility = layer_2(vol_net_layer_1)


loss = 0.5*tf.reduce_mean((volatility-volatility_)**2)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#train_step = tf.train.AdamOptimizer().minimize(loss)

vol_saver = tf.train.Saver()

reps = 10**5
batch_size = 15
check_every = 100

pred_losses = []
sample_losses = []

load = True

evaluate = True

if evaluate:
	with tf.Session() as vsess:
		saver.restore(vsess, "/volatility_0/model_most_recent.ckpt")

		R = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
		S = ['h','s','c','d']

		cards = []
		for r in R:
			for s in S:
				cards.append(r+s)

		holes = []
		for i, c1 in enumerate(cards):
			for c2 in cards:
				if c1 == c2 or sorted((c1, c2)) in holes:
					continue
				else:
					holes.append(sorted((c1, c2)))

		



		vecs = vectorize_hands(holes, [[] for _ in holes], normalize = False)

		vols = vsess.run(volatility, feed_dict = {card_vector_ : vecs})

		#print(g1, g2)
	vols = list(zip(vols, holes))

	best_holes = sorted(vols, reverse = False)
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
	
	plt.plot(xs, hist)
	plt.show()

	end#I dont want it to keep going



with tf.Session() as vsess:
	if load:
		saver.restore(vsess, "/volatility_0/model_most_recent.ckpt")
	else:
		vsess.run(tf.global_variables_initializer())
    
	for r in range(reps):

		if r % (reps//100) == 0:
			print(" ")
			print(datetime.datetime.now())
			print("training progress: "+str(r/reps))

		hand, vol = goodness_data_labels(batch_size)
		#print(hand.shape)
		#print(vol.shape)
		vsess.run(train_step, feed_dict = {card_vector_ : hand, volatility_ : vol})
		
		if r%check_every==0:

			current_prediction_loss = vsess.run(
				loss, feed_dict = {card_vector_ : hand, volatility_ : vol})

			pred_losses.append(current_prediction_loss)


			#save_path = saver.save(sess, "/volatility_0/model_rep_"+str(r)+".ckpt")
			save_path = saver.save(vsess, "/volatility_0/model_most_recent.ckpt")

			np.save("volatility_prediction_losses_1.npy", np.array(pred_losses))

		if r%(10)==0:
			hand, vol = goodness_data_labels(batch_size, custom_index = 536)
			#print(hand.shape)
			#print(vol.shape)
			sample_loss = vsess.run(loss, feed_dict = {card_vector_ : hand, volatility_ : vol})
			sample_losses.append(sample_loss)
			#print(sample_loss)


			
def granulate(list):
	return [np.mean(list[10*i:10*(i+1)]) for i in range(len(list)//10)]

np.save("volatility_prediction_losses_1.npy", np.array(pred_losses))

plt.plot(granulate(sample_losses))
plt.show()


plt.plot(pred_losses)
plt.show()
