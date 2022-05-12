from collections import namedtuple
instance = namedtuple("instance","W w A N")

def read(archive):
	f = open("kqbf/" + archive,'r')

	N = int(f.readline())
	W = int(f.readline())
	w = list(map(int,f.readline().split()))

	A = {}

	for i in range(N):
		line = list(map(int,f.readline().split()))

		for j in range(N-i):
			A[(i,j)] = line[j]

		for j in range(N-i,N):
			A[(i,j)] = 0

	return instance(W = W, w = w, A = A, N = N)
