from queue import PriorityQueue
from math import sqrt

class BestFirstSearch:

    def __init__(self, graph, map_letter_point):
        self.graph = graph
        self.map_letter_point = map_letter_point
        self.g = {}

    def initializePoint(self, source, destination):
        x = self.map_letter_point[source].x
        y = self.map_letter_point[source].y

        xi = self.map_letter_point[destination].x
        yi = self.map_letter_point[destination].y

        self.g[destination] = self.g[source] + sqrt((x - xi)**2 + (y - yi)**2)
    
    def revertPath(self, path):
        list = []
        currentPosition = 'target'

        while currentPosition != 'initial':
            list.append(currentPosition)
            currentPosition = path[currentPosition]

        list.append("initial")
        list.reverse()

        return list

    def run(self):
        priority_queue = PriorityQueue()
        priority_queue.put((0, "initial"))
        visited = ["initial"]
        self.g["initial"] = 0
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
                        self.initializePoint(point, key)
                        priority_queue.put((self.g[key], key))
                        
        if (flag):

            path = self.revertPath(path)
            distance = self.g["target"]
            return (visitations, distance, path)