import time
import random

# from archive
from python import read_input



# GLOBAL VARIABLES
A = None
N = None
W = None
w_ = None
TIME_LIMIT = 60*30


# TO IDENTIFY WHICH LOCAL SEARCH WERE CHOOSE
SEARCH_BEST_IMPROVING = 1
SEARCH_FIRST_IMPROVING = 2
SEARCH_DEFAULT = 3

# TO IDENTIFY WHICH CONSTRUCTIVE SEARCH WERE CHOOSE
CONSTRUCTIVE_DEFAULT = 1
CONSTRUCTIVE_RANDOM_PLUS_GREEDY = 2
CONSTRUCTIVE_SAMPLED_GREEDY = 3


class Solution:
    x = None
    cost = None
    constraint = None

    def __init__(self, x):
        if x is None:
            self.x = [1 if random.random() > 0.5 else 0 for i in range(N)]
            random.shuffle(self.x)
        else:
            self.x = x
        self.update_solution(self.x)

    def update_solution(self, x):
        self.x = x
        self.cost = 0
        for i in range(N):
            for j in range(N):
                self.cost += A[(i, j)] * self.x[i] * self.x[j]
        self.constraint = 0
        for i in range(N):
            self.constraint += w_[i] * x[i]
        self.make_feasible()

    def make_feasible(self):
        if self.constraint > W:
            i = random.randint(0, len(self.x) - 1)
            while self.x[i] == 0:
                i = (i + 1) % len(self.x)
            self.x[i] = 0
            self.update_solution(self.x)


class Grasp:
    best_solution = None
    time_limit = None
    start_time = None
    max_iterations = None
    alfa = None
    type_local_search = None
    type_constructive_method = None
    p = None

    def __init__(self, time_limit, max_iterations, type_local_search, type_constructive_method, alfa, p):
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.alfa = alfa
        self.type_local_search = type_local_search
        self.type_constructive_method = type_constructive_method
        self.p = p

    def execute(self):
        self.start_time = time.time()
        self.best_solution = Solution(None)
        while self.time_limit > (time.time() - self.start_time):
            current_solution = None

            if self.type_constructive_method == CONSTRUCTIVE_DEFAULT:
                current_solution = self.greedy_randomized_construction()
            elif self.type_constructive_method == CONSTRUCTIVE_SAMPLED_GREEDY:
                current_solution = self.constructive_heuristic_sampled_greedy()
            elif self.type_constructive_method == CONSTRUCTIVE_RANDOM_PLUS_GREEDY:
                current_solution = self.constructive_heuristic_random_plus_greedy()

            # we update the best_solution in local search
            if self.type_local_search == SEARCH_BEST_IMPROVING:
                self.local_search_first_improving(current_solution)
            elif self.type_local_search == SEARCH_BEST_IMPROVING:
                self.local_search_best_improving(current_solution)
            elif self.type_local_search == SEARCH_DEFAULT:
                self.local_search_default(current_solution)
        return self.best_solution

    def update_best_solution(self, solution):
        if solution.cost > self.best_solution.cost:
            self.best_solution = solution
            return True
        return False

    def local_search_first_improving(self, solution):
        for i in range(N):
            for j in range(i + 1, N):
                new_x = solution.x.copy()
                new_x[i], new_x[j] = new_x[j], new_x[i]
                if self.update_best_solution(Solution(new_x)):
                    return

    def local_search_best_improving(self, solution):
        for i in range(N):
            for j in range(i + 1, N):
                new_x = solution.x.copy()
                new_x[i], new_x[j] = new_x[j], new_x[i]
                self.update_best_solution(Solution(new_x))

    def local_search_default(self, solution):
        for _ in range(self.max_iterations):
            new_x = solution.x.copy()
            i = random.randint(0, len(solution.x) - 1)
            j = random.randint(0, len(solution.x) - 1)
            new_x[i], new_x[j] = new_x[j], new_x[i]
            self.update_best_solution(Solution(new_x))

    def greedy_randomized_construction(self):
        construction_x = []
        solution = Solution(None)
        search_space = [0, 1]

        for i in range(N):
            c_min, c_max, contribution = +1e11, -1e11, []
            # discover what is the contribution of element i in each solution
            for bit in search_space:
                sol_temp = Solution(solution.x[0:i] + [bit] + solution.x[i + 1:N])
                contribution.append(sol_temp.cost)
                c_min = min(c_min, sol_temp.cost)
                c_max = max(c_max, sol_temp.cost)

            rcl_indexes = []
            for j in range(len(search_space)):
                if contribution[j] >= (c_max - self.alfa * (c_max - c_min)):
                    rcl_indexes.append(j)

            if len(rcl_indexes) == 0:
                print("{} {} {}".format(c_max, c_min, contribution))
                exit(1)

            select_random_from_rcl = rcl_indexes[random.randint(0, len(rcl_indexes) - 1)]
            construction_x.append([0, 1][select_random_from_rcl])
        return Solution(construction_x)

    def constructive_heuristic_random_plus_greedy(self):
        construction_x = []
        solution = Solution(None)
        search_space = [0, 1]

        for i in range(N):
            contribution = []

            # first p steps, the solutions is generated randomly
            if self.p > i:
                construction_x.append(random.randint(0,1))
            else:
                # discover what is the contribution of element i in each solution
                for bit in search_space:
                    sol_temp = Solution(solution.x[0:i] + [bit] + solution.x[i + 1:N])
                    contribution.append(sol_temp.cost)

                bit = 1 if contribution[1] > contribution[0] else 0
                construction_x.append(bit)
        return Solution(construction_x)

    def constructive_heuristic_sampled_greedy(self):
        construction_x = []
        solution = Solution(None)
        search_space = [0, 1]

        for i in range(N):
            sum_constraint = []
            # discover what is the contribution of element i in each solution
            for bit in search_space:
                sol_temp = Solution(solution.x[0:i] + [bit] + solution.x[i + 1:N])
                sum_constraint.append(sum([sol_temp.x[i] * w_[i] for i in range(N)]))
            bit_greedy_evaluation = 1 if sum_constraint[1] < sum_constraint[0] else 0
            construction_x.append(bit_greedy_evaluation)
        return Solution(construction_x)

def main():

    archive_names = ["kqbf020", "kqbf040", "kqbf060", "kqbf080", "kqbf100", "kqbf200", "kqbf400"]

    for archive in archive_names:
        I = read_input.read(archive)
        global N, W, A, w_
        N = I.N
        A = I.A
        W = I.W
        w_ = I.w

        #(self, time_limit, max_iterations, type_local_search, type_constructive_method, alfa, p):
        grasp1 = Grasp(TIME_LIMIT, None, SEARCH_FIRST_IMPROVING, CONSTRUCTIVE_DEFAULT, 0.2, None).execute()

        #(self, time_limit, max_iterations, type_local_search, type_constructive_method, alfa, p):
        grasp2 = Grasp(TIME_LIMIT, N * N, SEARCH_DEFAULT, CONSTRUCTIVE_DEFAULT, 0.7, None).execute()

        #(self, time_limit, max_iterations, type_local_search, type_constructive_method, alfa, p):
        grasp3 = Grasp(TIME_LIMIT, None, SEARCH_BEST_IMPROVING, CONSTRUCTIVE_DEFAULT, 0.2, None).execute()

        #(self, time_limit, max_iterations, type_local_search, type_constructive_method, alfa, p):
        grasp4 = Grasp(TIME_LIMIT, N * N, SEARCH_DEFAULT, CONSTRUCTIVE_RANDOM_PLUS_GREEDY, None, N/2).execute()

        #(self, time_limit, max_iterations, type_local_search, type_constructive_method, alfa, p):
        grasp5 = Grasp(TIME_LIMIT, N * N, SEARCH_DEFAULT, CONSTRUCTIVE_SAMPLED_GREEDY, None, N/2).execute()

        print("{}".format(N))
        print("PADRÃO cost: {}".format(grasp1.cost))
        # print("x: {}".format(grasp1.x))

        print("PADRÃO+ALPHA cost: {}".format(grasp2.cost))
        # print("x: {}".format(grasp2.x))

        print("PADRÃO+BEST cost: {}".format(grasp3.cost))
        # print("x: {}".format(grasp3.x))

        print("PADRÃO+HC1 cost: {}".format(grasp4.cost))
        # print("x: {}".format(grasp4.x))

        print("PADRÃO+HC2 cost: {}".format(grasp5.cost))
        # print("x: {}".format(grasp5.x))
        print("\n\n")

if __name__ == "__main__":
    main()
