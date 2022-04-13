class point:
	x = None
	y = None

	def __init__(self, x, y):
		self.x = x
		self.y = y

	def is_equal(self, other_point):
		return self.x == other_point.x and self.y == other_point.y

class polygon:
	points = None
	number = None
	def __init__(self, number, points):
		self.number = number
		self.points = points


#####################################################################################
# geometric functions


'''
	dado que os pontos (p), (q) e (r) estão alinhados, queremos descobrir se o ponto (r) esta entre (p) e (q)
'''
def isOnSegment(p,q,r):
	if q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y):
		return True
	return False

'''
	qual a orientação do (ptc) em relação a (pta) e (ptb)
	0 estão alinhados
	1 esquerda
	-1 direita
'''
def orientation(pta, ptb, ptc):
	det = +(ptb.x*ptc.y) +(pta.y*ptc.x) +(pta.x*ptb.y) -(pta.y*ptb.x) -(pta.x*ptc.y) -(ptc.x*ptb.y)
	if det == 0:
		return 0
	if det > 0:
		return 1
	return -1

'''
	se o segmento (p1)-(q1) intersecta o segmento (p2)-(q2)
'''
def segments_intersect(p1, q1, p2, q2):
	o1 = orientation(p1, q1, p2)
	o2 = orientation(p1, q1, q2)
	o3 = orientation(p2, q2, p1)
	o4 = orientation(p2, q2, q1)
	if o1 != o2 and o3 != o4:
		return True
	if o1 == 0 and isOnSegment(p1, p2, q1):
		return True
	if o2 == 0 and isOnSegment(p1, q2, q1):
		return True
	if o3 == 0 and isOnSegment(p2, p1, q2):
		return True
	if o1 == 0 and isOnSegment(p2, q1, q2):
		return True
	return False


'''
	sentido: horario ou anti-horario!	
	retorno:
		0 se o ponto esta na borda
		1 se o ponto esta dentro
		-1 se o ponto esta fora
'''
def point_inside_polygon(p, polygon, map_letter_point):
	n = len(polygon.points)
	windingNumber = 0

	for i in range(n):
		point_i = map_letter_point.get(polygon.points[i])
		if p.is_equal(point_i):
			# print("DEBUG 1")
			return 0

		point_j = map_letter_point.get(polygon.points[(i+1)%n]) # next point!!!!!! 

		if point_i.y == p.y and point_j.y == p.y:
			if min(point_i.x, point_j.x) <= p.x and p.x <= max(point_i.x, point_j.x):
				return 0
		else:
			below = point_i.y < p.y

			if below != (point_j.y < p.y):
				ori = orientation(point_i,point_j,p)
				if ori == 0:
					return 0
				if below == (ori > 0):
					windingNumber += (1 if below else -1)

	return 1 if windingNumber%2 != 0 else -1


'''
	Dado um polygono -> lista de vertices, queremos saber se alguma das arestas intersecta o segmento (p)-(q)
	importante: não podemos considera intersecção entre vertices!
'''
def segment_intersect_polygon(p, q, polygon, map_letter_point):
	n = len(polygon.points)
	for i in range(n-1):
		point_i = map_letter_point.get(polygon.points[i])
		point_j = map_letter_point.get(polygon.points[(i+1)%n])
		if segments_intersect(p, q, point_i, point_j):
			if not p.is_equal(point_i) and not p.is_equal(point_j) and not q.is_equal(point_i) and not q.is_equal(point_j): 
				return True
	return False