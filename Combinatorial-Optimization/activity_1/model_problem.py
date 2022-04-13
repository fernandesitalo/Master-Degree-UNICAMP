import gurobipy as gp
from gurobipy import GRB
from utils_problem import *


def build_model_gurobi(instance):

	# Create a new model
	model = gp.Model()

	# Create variables
	X_plf = model.addVars(instance.P_plf.keys(), obj = instance.P_plf, vtype=GRB.CONTINUOUS, name='producao')
	Y_pfj = model.addVars(instance.t_pfj.keys(), obj = instance.t_pfj, vtype=GRB.CONTINUOUS, name='entrega')
	
	for j in range(instance.J):
		for p in range(instance.P):
			model.addConstr(instance.D_jp[j,p] <= Y_pfj.sum(p,'*',j), "(1)")

	for f in range(instance.F):
		for m in range(instance.M):
			model.addConstr(instance.R_mf[m,f] >= gp.quicksum(X_plf[p,l,f] * instance.r_mpl[m,p,l] for l in range(instance.L) for p in range(instance.P)), "(2)")

	for l in range(instance.L):
		for f in range(instance.F):
			model.addConstr(instance.C_lf[l,f] >= X_plf.sum('*',l,f), "(3)")

	for p in range(instance.P):
		for f in range(instance.F):
				model.addConstr(X_plf.sum(p,'*',f) == Y_pfj.sum(p,f,'*'), "(4)")

	return model
