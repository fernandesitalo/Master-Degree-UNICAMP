from gurobipy import GRB


'''
	retorna a quantidade de horas em segundos
'''
def QTD_HRS(h):
	return 60 * 60 * h

def print_solution(X_plf, P_plf, Y_pfj, t_pfj):
	print("==================== solution ========================")
	print("X_plf: {}".format(X_plf))
	print("Y_pfj: {}".format(Y_pfj))
