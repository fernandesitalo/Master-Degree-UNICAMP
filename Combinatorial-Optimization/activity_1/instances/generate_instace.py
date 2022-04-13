import random

for ii in range(1,10+1):
	for jj in range(1,10+1):
		g = open("instance_"+str(ii*100)+"_"+str(jj),'w') #instance_i_j

		J = 100*ii
		F = random.randint(J,2*J)
		L = random.randint(5,10)
		M = random.randint(5,10)
		P = random.randint(5,10)

		g.write(str(J) + "\n")
		g.write(str(F) + "\n")
		g.write(str(L) + "\n")
		g.write(str(M) + "\n")
		g.write(str(P) + "\n")

		for j in range(J):
			for p in range(P):
				g.write(str(random.randint(10,20)) + "\n")

		for m in range(M):
			for p in range(P):
				for l in range(L):
					g.write(str(random.randint(1,5)) + "\n")

		for m in range(M):
			for f in range(F):
				g.write(str(random.randint(800,1000)) + "\n")

		for l in range(L):
			for f in range(F):
				g.write(str(random.randint(80,100)) + "\n")

		for p in range(P):
			for l in range(L):
				for f in range(F):
					g.write(str(random.randint(10, 100)) + "\n")

		for p in range(P):
			for f in range(F):
				for j in range(J):
					g.write(str(random.randint(10, 20)) + "\n")
