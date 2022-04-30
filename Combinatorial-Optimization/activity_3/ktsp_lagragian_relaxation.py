import sys
import math
import random
import time
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB

# global variables
N,K = None, None
START_TIME = None

CHECK_SOLUTION = False
TIME_LIMIT = 30*60

def subtourelim_updated(model, where):
    subtourelim1(model, where)
    subtourelim2(model, where)


# Callback - use lazy constraints to eliminate sub-tours
def subtourelim1(model, where):
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._x1)
        # find the shortest cycle in the selected edge list
        tour = subtour(vals)
        if len(tour) < N:
            # add subtour elimination constr. for every pair of cities in tour
            model.cbLazy(gp.quicksum(model._x1[i, j] for i, j in combinations(tour, 2)) <= len(tour)-1)


# Callback - use lazy constraints to eliminate sub-tours
def subtourelim2(model, where):
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._x2)
        # find the shortest cycle in the selected edge list
        tour = subtour(vals)
        if len(tour) < N:
            # add subtour elimination constr. for every pair of cities in tour
            model.cbLazy(gp.quicksum(model._x2[i, j] for i, j in combinations(tour, 2)) <= len(tour)-1)


# Given a tuplelist of edges, find the shortest subtour
def subtour(vals):
    # make a list of edges selected in the solution
    edges = gp.tuplelist((i, j) for i, j in vals.keys() if vals[(i, j)] > 0.5)
    unvisited = list(range(N))
    cycle = range(N+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle


def read_file(filename, n_points):
    arq = open(filename,"r")
    points1, points2 = [],[]
    for _ in range(n_points):
        x1,y1,x2,y2 = map(int,arq.readline().split())
        points1.append((x1,y1))
        points2.append((x2,y2))
    return points1, points2


def calc_solution(tour_tsp1, tour_tsp2, ce1, ce2):
    cost_tsp2 = sum(\
            [ce2[max(tour_tsp2[i],tour_tsp2[(i+1)%N]), min(tour_tsp2[i],tour_tsp2[(i+1)%N])]\
            for i in range(N)])

    cost_tsp1 = sum(\
            [ce1[max(tour_tsp1[i],tour_tsp1[(i+1)%N]), min(tour_tsp1[i],tour_tsp1[(i+1)%N])]\
            for i in range(N)])

    return cost_tsp1 + cost_tsp2


def print_solution(model, dist1, dist2, edges1, edges2):
    tour_tsp1 = subtour(model.getAttr('X', edges1))

    print("len(tour_tsp1): {}     N: {}".format(len(tour_tsp1), N))

    if CHECK_SOLUTION == True:
        assert len(tour_tsp1) == N

    print('\n')
    print('TSP_1:')
    print('Optimal tour tsp1: %s' % str(tour_tsp1))
    print('Optimal cost tsp1: {}'.format(\
        sum(\
            [dist1[max(tour_tsp1[i],tour_tsp1[(i+1)%N]), min(tour_tsp1[i],tour_tsp1[(i+1)%N])]\
            for i in range(N)])))
    print('')

    tour_tsp2 = subtour(model.getAttr('X', edges2))

    if CHECK_SOLUTION == True:
        assert len(tour_tsp2) == N

    print('TSP_2:')
    print('Optimal tour tsp2: %s' % str(tour_tsp2))
    print('Optimal cost tsp2: {}'.format(\
        sum(\
            [dist2[max(tour_tsp2[i],tour_tsp2[(i+1)%N]), min(tour_tsp2[i],tour_tsp2[(i+1)%N])]\
            for i in range(N)])))
    print('')

    count_k = 0

    for i in range(N):
        next_i = (i+1)%N
        edge1 = [tour_tsp1[i], tour_tsp1[next_i]]
        edge1.sort()

        for j in range(N):
            next_j = (j+1)%N
            edge2 = [tour_tsp2[j], tour_tsp2[next_j]]
            edge2.sort()

            if edge1 == edge2:
                count_k += 1

    if CHECK_SOLUTION == True:
        assert count_k >= K

    print("count_k: {} checked! =)".format(count_k))
    print("Total optimal cost: {}".format(model.ObjVal))


def run_model(lagrange_multiplier, ce1, ce2):
    # print("\n>> run model with lagrange_multiplier: {}\n".format(lagrange_multiplier))

    m = gp.Model()
    m.setParam('TimeLimit', TIME_LIMIT - (time.time() - START_TIME)) # in seconds

    # Create variables
    edges1 = m.addVars(ce1.keys() , vtype=GRB.BINARY, name='edges1') #, obj=dist1'''
    for i,j in edges1.keys():
        edges1[j,i] = edges1[i,j]

    edges2 = m.addVars(ce2.keys() , vtype=GRB.BINARY, name='edges2') #, obj=dist2'''
    for i,j in edges2.keys():
        edges2[j,i] = edges2[i,j]

    # just getting the indices from edges - can be dist1.keys() or dist2.keys()
    ze = m.addVars(ce1.keys(), vtype=GRB.BINARY, name='duplication')

    m.addConstrs(edges1.sum(i, '*') == 2 for i in range(N))
    m.addConstrs(edges2.sum(i, '*') == 2 for i in range(N))

    # m.addConstr(ze.sum() >= K) #constraint that has been relaxed

    for i in range(N):
        for j in range(i): 
            # m.addConstr(edges1[i,j] + edges2[i,j] >= 2 * duplication[i,j])
            m.addConstr(edges1[i,j] >= ze[i,j])
            m.addConstr(edges2[i,j] >= ze[i,j])

     # Optimize model
    m._x1 = edges1
    m._x2 = edges2
    m._ze = ze # testar isso depois
    m.Params.LazyConstraints = 1


    x1 = m._x1
    x2 = m._x2
    ze = m._ze

    m.setObjective( gp.quicksum( x1[i,j]*ce1[i,j] + x2[i,j]*ce2[i,j] for i in range(N) for j in range(i)) \
                        + lagrange_multiplier * (K - gp.quicksum(ze[i,j] for i in range(N) for j in range(i))) , GRB.MINIMIZE)

    start_time = time.time()
    m.optimize(subtourelim_updated)
    end_time = time.time()

    print("Seconds to run the model: {}\n".format((end_time - start_time)))
    
    return m


def calc_dists(points1, points2):
    # Dictionary of Euclidean distance between each pair of points
    dist1 = {(i, j):
            math.ceil(math.sqrt(sum((points1[i][w]-points1[j][w])**2 for w in range(2))))
            for i in range(N) for j in range(i)}
    
    dist2 = {(i, j):
            math.ceil(math.sqrt(sum((points2[i][w]-points2[j][w])**2 for w in range(2))))
            for i in range(N) for j in range(i)}

    return dist1, dist2


def heuristics_upper_bound2(error, model, ce1, ce2):
    tour_tsp1 = subtour(model.getAttr('X', model._x1))
    tour_tsp2 = subtour(model.getAttr('X', model._x2))

    assert len(tour_tsp1) == N
    assert len(tour_tsp2) == N

    l, r = 0, -1
    window_cost = 0
    minimum_window_cost = 1e10
    min_l, min_r = 0,-1

    while N > r+1:
        while K+1 > (r-l+1) and N > r+1:
            r += 1
            cost_edge1 = ce2[max(tour_tsp1[r],tour_tsp1[(r+1)%N]),min(tour_tsp1[r],tour_tsp1[(r+1)%N])]
            window_cost += cost_edge1

        if minimum_window_cost > window_cost:
            minimum_window_cost = window_cost
            min_l = l
            min_r = r

        cost_edge1 = ce2[max(tour_tsp1[l],tour_tsp1[(l+1)%N]), min(tour_tsp1[l],tour_tsp1[(l+1)%N])]
        window_cost -= cost_edge1
        l += 1

    vertex_not_in_tsp2 = set(i  for i in range(N))
    tsp2 = []
    while min_r >= min_l:
        vertex_not_in_tsp2.remove(tour_tsp1[min_l])
        tsp2.append(tour_tsp1[min_l])
        min_l += 1


    while len(tsp2) != N:
        current_vertex = tsp2[-1]

        next_vertex, cost_next_vertex = -1, 1e10
        for v in vertex_not_in_tsp2:
            if cost_next_vertex > ce2[(max(current_vertex, v),min(current_vertex, v))] and random.random():
                cost_next_vertex = ce2[(max(current_vertex, v),min(current_vertex, v))]
                next_vertex = v

        assert next_vertex != -1

        vertex_not_in_tsp2.remove(next_vertex)
        tsp2.append(next_vertex)


    assert len(set(tsp2)) == N

    # just to check if the tsp has K commom edges
    count_k = 0
    for i in range(N):
        next_i = (i+1)%N
        edge1 = [tour_tsp1[i], tour_tsp1[next_i]]
        edge1.sort()

        for j in range(N):
            next_j = (j+1)%N
            edge2 = [tsp2[j], tsp2[next_j]]
            edge2.sort()

            if edge1 == edge2:
                count_k += 1

    print("count_k: {}".format(count_k))
    assert count_k >= K

    return calc_solution(tour_tsp1, tsp2, ce1, ce2)


def subgradient_method(ce1, ce2):

    # initial mulpitlier lagrangean
    lagrange_multiplier = 0
    Z_ub, Z_lb = 1e10, -1e10
    model = None
    pi = 2.0
    while True:

        current_time = time.time()

        if (current_time - START_TIME) > TIME_LIMIT:
            break

        model = run_model(lagrange_multiplier, ce1, ce2)
        
        ze = model.getAttr('X', model._ze)
        penalidade = K - sum(ze[i,j] for i in range(N) for j in range(i)) # for all relaxed restrictions
        
        Z_lb_current = -1e10

        if 0 >= penalidade: 
            Z_lb = max(model.ObjVal, Z_lb)
        else:
            Z_ub = min( heuristics_upper_bound2(math.ceil(penalidade), model, ce1, ce2) , Z_ub)

        print("\n\n=============================================================")
        print("Z_lb: {} Z_ub: {}".format(Z_lb, Z_ub))
        print("lagrange_multiplier: {}".format(lagrange_multiplier))
        print("penalidade: {}".format(penalidade))
        print("=============================================================\n\n")

        if penalidade == 0.0 or Z_lb == Z_ub:
            break

        alfa = pi * (Z_ub - Z_lb) / (penalidade**2)
        pi *= 0.95
        lagrange_multiplier += alfa * penalidade
        lagrange_multiplier = max(0, lagrange_multiplier)

    return model, Z_ub


def main():
    # Parse argument
    if len(sys.argv) < 4:
        print('Usage: tsp.py npoints k_paramter name_file')
        sys.exit(1)

    global N, K, START_TIME
    N = int(sys.argv[1])
    K = int(sys.argv[2])
    name_file = str(sys.argv[3])

    print("n = {}".format(N))
    print("k = {}".format(K))

    points1, points2 = read_file(name_file, N)
    dist1, dist2 = calc_dists(points1, points2)

    START_TIME = time.time()
    m,solution = subgradient_method(dist1, dist2)

    end_time = time.time()

    print("Seconds to run the method: {}".format(end_time - START_TIME))

if __name__ == "__main__":
    main()