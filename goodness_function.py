import numpy as np 
import matplotlib.pyplot as plt 
from tools import CARD_VECTOR_DICT 

def myhash(string):
	out = 1

	for s in string:
		n = ord(s)
		n = (n+ (n%3)*n**2 + (n%4)*n**3)

		out*=n 

	return out%100


def goodness(hole, community):
	

	#for now its arbitrary
	n1 = myhash(hole[0]+hole[1])%100

	for c in community:
		temp = (myhash(c)%10)-5
		n1+=temp

	n1-=50
	n1/=100

	return n1

