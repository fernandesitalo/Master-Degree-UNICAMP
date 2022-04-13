from read_and_process_input import *
from geometric_functions import *
from IdaStar import IdaStar
from IterativeDeepening import ID
from BestFirstSearch import BestFirstSearch
from AStar import AStar

def main():
	archive = "input.txt"
	obj_input = read_and_process_input(archive)
	print(obj_input.graph)
	map_letter_point = obj_input.get_map_letter_point()

	bestFS = BestFirstSearch(obj_input.graph, map_letter_point)
	visitations, distance, path = bestFS.run()
	print("Best First Search")
	print("visitations:", visitations)
	print("cost:", distance)
	print("path:", path)
	print("")

	id = ID(obj_input.graph, map_letter_point)
	visitations, distance, path = id.run()
	print("IterativeDeepening")
	print("visitations:", visitations)
	print("cost:", distance)
	print("path:", path)
	print("")

	aStar = AStar(obj_input.graph, map_letter_point)
	visitations, distance, path = aStar.run()
	print("A*")
	print("visitations:", visitations)
	print("cost:", distance)
	print("path:", path)
	print("")

	idaStar = IdaStar(obj_input.graph, map_letter_point)
	visitations, distance, path = idaStar.run()
	print("IDA*")
	print("visitations:", visitations)
	print("cost:", distance)
	print("path:", path)
	print("")


if __name__ == '__main__':
    main()