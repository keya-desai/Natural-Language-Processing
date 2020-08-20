import numpy as np 
from collections import Counter
from collections import defaultdict

def compute_q_mle(t1, t2):

	q_mle = Counter()
	root = Counter()
	binary_rules = defaultdict(list)
	binary_rules_list = []

	for rule in t1:
		root[(rule[0])] += 1
		q_mle[rule] += 1
		if(len(rule) == 3):
			binary_rules_list.append(rule)
			if rule[0] not in binary_rules[(rule[1], rule[2])]:
				binary_rules[(rule[1], rule[2])].append(rule[0])

	for rule in t2:
		root[(rule[0])] += 1
		q_mle[rule] += 1
		if(len(rule) == 3):
			binary_rules_list.append(rule)
			if rule[0] not in binary_rules[(rule[1], rule[2])]:
				binary_rules[(rule[1], rule[2])].append(rule[0])


	for item in q_mle.items():
		q_mle[item[0]] = item[1]/root[item[0][0]]

	return q_mle, root.keys(), binary_rules, set(binary_rules_list)


def inner_terms(x, potential, states, binary_rules):

	# init
	alpha = defaultdict(list)
	words = x.split()
	n = len(words)

	for i in range(1, n+1):
		for a in states:
			if(potential[(a, words[i-1])]):
				alpha[(i,i)].append((a, potential[(a, words[i-1])]))
				

	for length in range(1, n):
		for i in range(1, n-length + 1):
			j = i + length
			for k in range(i, j):
				# print("(%d,%d), (%d,%d)" % (i, k, k+1, j))

				if((i,k) in alpha):
					b = alpha[(i, k)][0][0]
					left_prob = alpha[(i, k)][0][1]
				else:
					continue

				if((k+1,j) in alpha):
					c = alpha[(k+1, j)][0][0]
					right_prob = alpha[(k+1, j)][0][1]
				else:
					continue

				if((b,c) in binary_rules):
					a = binary_rules[(b,c)][0]
				else:
					continue

				if(len(alpha[(i,j)]) != 0 and alpha[(i,j)][0][0] == a):
					alpha[(i,j)] = (a, alpha[(i,j)][0][1] + potential[(a,b,c)] * right_prob * left_prob)
				else:
					alpha[(i,j)].append((a, potential[(a,b,c)] * right_prob * left_prob) )

	return alpha


def outer_terms(x, potential, states, binary_rules_set, alpha, debug = False):
	beta = dict()
	n = len(x.split())

	# intialization
	for s in states:
		if(s != 'S'):
			beta[(s,1,n)] = 0
		else:
			beta[(s,1,n)] = 1
	
	# dictionary of rules in which A is left child
	left_rules = defaultdict(list)

	# dictionary of rules in which A is right child
	right_rules = defaultdict(list)

	for a in states:
		for rule in binary_rules_set:
			if(a == rule[1]):
				left_rules[a].append({"b" : rule[0], "c" : rule[2]})
			if(a == rule[2]):
				right_rules[a].append({"b" : rule[0], "c" : rule[1] })
	if(debug):	print("left rules :\n", left_rules)
	if(debug): 	print("right rules :\n", right_rules)

	for length in range(n-1, -1, -1):
		for i in range(n-length , 0, -1):
			j = i + length
			if(debug): print("\n",i,j)
			if(i == 1 and j == n):
				continue
			for a in states:
				summ = 0
				for bc in right_rules[a]:
				# sum of trees with a = left child
					b = bc['b']
					c = bc['c']
					for k in range(1, i):
						if(debug):	print("a = right child")
						if(debug):	print(a, (b,c,a), (c,k,i-1), (b,k,j))
						if((c,k,i-1) in alpha):
							summ += potential[(b,c,a)] * alpha[(c,k,i-1)] * beta[(b,k,j)] 
							if(debug):	print(potential[(b,c,a)] , alpha[(c,k,i-1)],  beta[(b,k,j)] )

				# sum of trees with a = right child
				for bc in left_rules[a]:
					
					b = bc['b']
					c = bc['c']
					for k in range(j+1, n+1):
						if(debug):	print("a = left child")
						if(debug):	print(a, (b,a,c), (c,j+1,k), (b,i,k))
						if((c,j+1,k) in alpha):
							summ += potential[(b,a,c)] * alpha[(c,j+1,k)] * beta[(b,i,k)] 
							if(debug):	print(potential[(b,a,c)] * alpha[(c,j+1,k)] * beta[(b,i,k)] )
							

				beta[(a,i,j)] = summ

	if(debug):
		for item in beta.items():
			if(item[1]!=0):
				print(item)

	return beta



def inside_outside(x, t1, t2):

	q_mle, states, binary_rules, binary_rules_set = compute_q_mle(t1, t2)
	print("q_mle = ", q_mle)

	# for pcfg
	potential = q_mle 
	alpha_ = inner_terms(x, potential, states, binary_rules)
	alpha = dict()
	# changing form of alpha to [(a,i,j)] as key:
	for item in alpha_.items():
		alpha[(item[1][0][0],item[0][0],item[0][1])] = item[1][0][1]

	print("\nalpha :")
	for item in alpha.items():
			if(item[1]!=0):
				print(item)

	beta = outer_terms(x, potential, states, binary_rules_set, alpha)

	print("\nbeta :")
	for item in beta.items():
			if(item[1]!=0):
				print(item)
	
	mu = dict()
	n = len(x.split())
	for a in states:
		for i in range(1,n+1):
			for j in range(i,n+1):
				if((a,i,j) in alpha and (a,i,j) in beta):
					mu[(a,i,j)] = alpha[(a,i,j)] * beta[(a,i,j)]
				else:
					mu[(a,i,j)] = 0

	print("\nmu :")
	for item in mu.items():
			if(item[1]!=0):
				print(item)

	return mu



if __name__ == "__main__":

	x = "the boy saw the man with a telescope"

	t1 = [('S', 'NP', 'VP'), ('NP' , 'D', 'N'), ('D', "the"), ('N', "boy"), ('VP' , 'V', 'NP'),
			('V', "saw"), ('NP' , 'NP', 'PP'), ('NP' , 'D', 'N'), ('D', "the"), ('N', "man"),
			('PP' , 'P', 'NP'), ('P', "with"),('NP' , 'D', 'N'), ('D', "a"), ('N', "telescope")]

	t2 = [('S', 'NP', 'VP'), ('NP' , 'D', 'N'), ('D', "the"), ('N', "boy"), ('VP' , 'VP', 'PP'),
			('VP' , 'V', 'NP'), ('V', "saw"), ('NP' , 'D', 'N'), ('D', "the"), 
			('N', "man"), ('PP' , 'P', 'NP'), ('P', "with"),('NP' , 'D', 'N'), ('D', "a"), ('N', "telescope")]

	
	mu = inside_outside(x, t1, t2)

	print("\nmu['NP', 4, 8] = ", mu['NP', 4, 8])

	print("mu['VP', 3, 5] = ", mu['VP', 3, 5])
