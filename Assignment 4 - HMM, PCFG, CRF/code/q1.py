import numpy as np
from collections import Counter


def emission_probability(X, Y):
	t = Counter()
	state_count = Counter()
	o = Counter()
	
	for x,y in zip(X,Y):
		s = '*'
		for x_,s_ in zip(x.split(),y):
			t[(s_,s)] += 1
			state_count[s_] += 1
			o[(x_,s_)] += 1
			s = s_

		t[("STOP",s_)] += 1
		state_count["STOP"] += 1
		state_count['*'] += 1

	for item in o.items():
		s = item[0][1]
		val = item[1]/state_count[s]
		o[item[0]] = val	

	for item in t.items():
		s = item[0][1]
		val = item[1]/state_count[s]
		t[item[0]] = val	
	
	return t, o

def forward_backward(x, y, S, t, o):

	words = x.split()
	alpha = dict()
	beta = dict()
	mu = dict()
	m = len(y)

	for i, s in enumerate(S):
		alpha[(1,s)] = t[(s,'*')] * o[(words[0], s)]
		beta[(m,s)] = t[("STOP", y[i])]

	for j in range(2,len(y)+1):
		for s in S:
			summ = 0
			for s_ in S:
				summ += alpha[(j-1,s_)] * t[(s,s_)] * o[(words[j-1], s)]
			alpha[(j,s)] = summ
	# print("\nalpha = ")
	# print(alpha)

	for j in range(m-1,0,-1):
		for s in S:
			summ = 0
			for s_ in S:
				summ += beta[(j+1,s_)] * t[(s_,s)] * o[(words[j]),s_]
			beta[(j,s)] = summ

	# print("\nbeta = ")
	# print(beta)

	for j in range(1,m+1):
		for a in S:
			mu[(j,a)] = alpha[(j,a)] * beta[(j,a)]

	# print("\nmu =")
	# print(mu)

	return mu


if __name__ == "__main__":
	X = ["the man saw the cut", "the saw cut the man", "the saw"]
	Y = ["DNVDN", "DNVDN", "NN"]
	S = ['D', 'N', 'V']

	t, o = emission_probability(X,Y)
	# print(o)
	# print(t)

	for item in o.items():
		print(item)
	for item in t.items():
		print(item)

	mu = forward_backward(X[1], Y[1], S, t, o)
	print("\nprobability under the HMM that the 3rd is tagged with V wrt x2= ")
	print(mu[(3,'V')])

	mu = forward_backward(X[0], Y[0], S, t, o)
	print("\nprobability under the HMM that the 5th is tagged with N wrt x1= ")
	print(mu[(5,'N')])
