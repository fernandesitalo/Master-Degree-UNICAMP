from file_problem import *
from utils_problem import *
from model_problem import *
import time
import sys

PREFIX_PATH = "instances/"


def main():
	param = sys.argv[1:]
	archive = param[0]
	instance_problem = read_file_instance(PREFIX_PATH+archive)
	model = build_model_gurobi(instance_problem)
	model.Params.timelimit = 60*30
	start_time = time.time()
	model.optimize()
	end_time = time.time()
	print("Seconds to run the model: {}\n".format((end_time - start_time)))

if __name__ == "__main__":
    main()