from collections import namedtuple

instance = namedtuple("instance","J L F M P D_jp r_mpl R_mf C_lf P_plf t_pfj")

def read_file_instance(filename):
	
	arq = open(filename,"r")

	J = int(arq.readline())
	F = int(arq.readline())
	L = int(arq.readline())
	M = int(arq.readline())
	P = int(arq.readline())

	print("J: {}".format(J))
	print("F: {}".format(F))
	print("L: {}".format(L))
	print("M: {}".format(M))
	print("P: {}".format(P))

	print("\n\n")

	# demanda do cliente j, em toneladas, do produto p
	D_jp = {}
	for j in range(J):
		for p in range(P):
			D_jp[(j,p)] = int(arq.readline())
	
	# quantidade de matéria-prima m, em toneladas, necessária para produzir uma tonelada do produto p na máquina l;
	r_mpl = {}
	for m in range(M):
		for p in range(P):
			for l in range(L):
				r_mpl[(m,p,l)] = int(arq.readline())

	# quantidade de matéria-prima m, em toneladas, disponível na fábrica f;
	R_mf = {}	
	for m in range(M):
		for f in range(F):
			R_mf[(m,f)] = int(arq.readline())

	# capacidade disponível de produção, em toneladas, da máquina l na fábrica f
	C_lf = {}
	for l in range(L):
		for f in range(F):
			C_lf[(l,f)] = int(arq.readline())

	# custo de produção por tonelada do produto p utilizando a máquina l na fábrica f
	P_plf = {}
	for p in range(P):
		for l in range(L):
			for f in range(F):
				P_plf[(p,l,f)] = int(arq.readline())

	# custo de transporte por tonelada do produto p partindo da fábrica f até o cliente j
	t_pfj = {}
	for p in range(P):
		for f in range(F):
			for j in range(J):
				t_pfj[(p,f,j)] = int(arq.readline())

	return instance(J=J, L=L, F=F, M=M, P=P, D_jp=D_jp, r_mpl=r_mpl, R_mf=R_mf, C_lf=C_lf, P_plf=P_plf, t_pfj=t_pfj)
