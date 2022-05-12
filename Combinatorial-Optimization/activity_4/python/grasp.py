import time
import random

# from archive
from python import read_input

A = None
N = None
W = None
w_ = None


class Solution:
	x = None
	cost = None
	constraint = None

	def __init__(self, x):
		if x is None:
			self.x = [ 1 if random.random() > 0.5 else 0 for i in range(N)]
			random.shuffle(self.x)
		else:
			self.x = x
		self.update_solution(self.x)

	def update_solution(self, x):
		self.x = x
		self.cost = 0
		for i in range(N):
			for j in range(N):
				self.cost += A[(i,j)] * self.x[i] * self.x[j]

		self.constraint = 0

		for i in range(N):
			self.constraint += w_[i] * x[i]

		self.deixa_factivel()

	def deixa_factivel(self):
		while self.constraint > W:
			for i in range(N):
				if self.x[i] == 1:
					self.x[i] = 0
					self.update_solution(self.x)
					break


class Grasp:
	best_solution = None
	time_limit = None
	start_time = None
	max_iterations = None
	alfa = None

	def __init__(self, time_limit, max_iterations, alfa):
		self.time_limit = time_limit
		self.max_iterations = max_iterations
		self.alfa = alfa

	def execute(self):
		self.start_time = time.time()
		self.best_solution = Solution(None)
		while self.time_limit > (time.time() - self.start_time):
			current_solution = self.greedy_randomized_construction()
			# already update the best_solution
			self.local_search_default(current_solution)
		return self.best_solution

	def update_best_solution(self, solution):
		if solution.cost > self.best_solution.cost:
			self.best_solution = solution

	def local_search_default(self, solution):
		for i in range(N):
			for j in range(i+1,N):
				new_x = solution.x.copy()
				new_x[i], new_x[j] = new_x[j], new_x[i]
				self.update_best_solution(Solution(new_x))

	def build_RCL(self, solution):
		rcl = []
		for i in range(N):
			new_x = [solution.x[i]] + solution.x[0:i] + solution.x[i+1:N]
			rcl.append(Solution(new_x))
		return rcl

	def greedy_randomized_construction(self):


		contruction_x = []
		solution = Solution(None)
		rcl = self.build_RCL(solution)

		for i in range(N):
			c_min = +1e11
			c_max = -1e11
			contribution = []
			# discover what is the contribution of element i in each solution
			for sol in rcl:
				sol_temp = Solution(sol.x[0:i] + [solution.x[i]] + sol.x[i + 1:N])
				contribution.append(sol_temp.cost)
				c_min = min(c_min, sol_temp.cost)
				c_max = max(c_max, sol_temp.cost)

			rcl_indexs = []
			for j in range(N):
				# print("{}  {}".format(contribution[j], (c_max - self.alfa * (c_max - c_min))))
				if contribution[j] >= (c_max - self.alfa * (c_max - c_min)):
					rcl_indexs.append(j)

			if len(rcl_indexs) == 0:
				print("{} {} {}".format(c_max, c_min, contribution))
				exit(1)

			select_random_from_rcl = rcl_indexs[random.randint(0,len(rcl_indexs)-1)]
			contruction_x.append(solution.x[select_random_from_rcl])
		'''
			gero uma solucao
			troco a primeira posicao com todas e gero meu RCL
			para solcao do RCL faco:
				considero o elemento I da solucao inicial, na solcao atual do RCL
				pego os qe deram melhores resultados e escoho um randomicamente
				pego a I dessa solucao do RCL
		'''

		return Solution(contruction_x)


# def local_search_first_improving(self):
	# 	pass
	#
	# def local_search_best_improving(self):
	# 	pass
	#
	# def constructive_heuristic_ALTERNATIVO1(self):
	# 	pass
	#
	# def constructive_heuristic_ALTERNATIVO2(self):
	# 	pass



def main():
	archive = "kqbf040"
	I = read_input.read(archive)
	global N,W,A,w_
	N = I.N
	A = I.A
	W = I.W
	w_ = I.w

	print("{} {}".format(N,W))

	grasp = Grasp(60*7, 20, 0.2)

	print(grasp.execute().cost)

if __name__ == "__main__":
	main()