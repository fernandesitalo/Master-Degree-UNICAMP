from queue import PriorityQueue
from math import sqrt

class AStar:

    def __init__(self, graph, map_letter_point):
        self.graph = graph
        self.map_letter_point = map_letter_point
        self.f = {}
        self.g = {}

    def initializePoints(self, source, destination):

        x = self.map_letter_point[destination].x
        y = self.map_letter_point[destination].y

        xi = self.map_letter_point[source].x
        yi = self.map_letter_point[source].y

        xf = self.map_letter_point["target"].x
        yf = self.map_letter_point["target"].y

        if (destination != "initial"):
            self.g[destination] = self.g[source] + sqrt((x - xi)**2 + (y - yi)**2)
        else:
            self.g[destination] = 0

        self.f[destination] = self.g[destination] + sqrt((x - xf)**2 + (y - yf)**2)
    
    def revertPath(self, path):
        list = []
        distance = 0
        currentPosition = 'target'

        while currentPosition != 'initial':
            list.append(currentPosition)

            x1 = self.map_letter_point[currentPosition].x
            y1 = self.map_letter_point[currentPosition].y
            x2 = self.map_letter_point[path[currentPosition]].x
            y2 = self.map_letter_point[path[currentPosition]].y

            distance += sqrt((x1 - x2)**2 + (y1 - y2)**2)
            currentPosition = path[currentPosition]

        list.append("initial")
        list.reverse()

        return (list, distance)

    def run(self):
        priority_queue = PriorityQueue()
        priority_queue.put((0, "initial"))
        visited = ["initial"]
        self.initializePoints("initial", "initial")
        visitations = 0
        path = {}
        flag = False

        while (not priority_queue.empty()):
            visitations += 1
            point = priority_queue.get()[1]

            if point == "target":
                flag = True
                break

            else:
                for key in self.graph[point]:
                    if key not in visited:
                        visited.append(key)
                        path[key] = point
                        self.initializePoints(point, key)
                        priority_queue.put((self.f[key], key))
        
        if (flag):
            path, distance = self.revertPath(path)
            return (visitations, distance, path)