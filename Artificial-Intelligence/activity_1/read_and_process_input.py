from geometric_functions import *

class input_problem:
	polygons = None
	map_letter_point = None
	graph = None

	def __init__(self, polygons, map_letter_point):
		self.map_letter_point = map_letter_point
		self.polygons = polygons
		self.graph = {}
		self.make_graph()

	def make_graph(self):
		letters = list(self.map_letter_point.keys())
		n = len(letters)

		for letter in letters:
			self.graph[letter] = []
		for i in range(n):
			for j in range(i+1,	n):				
				point_i = self.map_letter_point.get(letters[i])
				point_j = self.map_letter_point.get(letters[j])
				if self.check_connectivity(point_i, point_j):
					self.graph[letters[i]].append(letters[j])
					self.graph[letters[j]].append(letters[i])


	def check_connectivity(self, point_a, point_b):
		is_intersect = False
		for a_polygon in self.polygons:
			if segment_intersect_polygon(point_a, point_b, a_polygon, self.map_letter_point):
				is_intersect = True
				break
		if is_intersect == True:
			return False
		midpoint_ab = point((point_a.x+point_b.x)/2.0,(point_a.y+point_b.y)/2.0)
		for a_polygon in self.polygons:
			if point_inside_polygon(midpoint_ab, a_polygon, self.map_letter_point) > 0:
				return False
		return True

	def get_map_letter_point(self):
		return self.map_letter_point

def read_and_process_input(archive_name):
	f = open(archive_name,'r')

	amount_of_points = int(f.readline())
	map_letter_point = {}
	for _ in range(amount_of_points):
		letter, point_x, point_y = map(str,f.readline().split())
		map_letter_point[letter] = point(float(point_x), float(point_y))

	polygons = []

	amount_of_polygons = int(f.readline())
	for _ in range(amount_of_polygons):
		input_polygon = [] 
		input_polygon = f.readline().split()
		number = input_polygon.pop(0)
		polygons.append(polygon(int(number), input_polygon))

	x_initial, y_initial = map(float,f.readline().split())
	x_target, y_target = map(float,f.readline().split())
	
	initial = point(x_initial, y_initial) # this vertex I will call of "initial"
	target = point(x_target, y_target) # this vertex I will call of "target"

	map_letter_point["initial"] =  initial
	map_letter_point["target"] = target

	return input_problem(polygons, map_letter_point)