import numpy as np 

R = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
S = ['h','s','c','d']

CARD_VECTOR_DICT = dict()

count = 0
for r in R:
	for s in S:
		CARD_VECTOR_DICT[r+s] = count
		count+=1
	