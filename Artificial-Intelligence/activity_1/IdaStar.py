class IdaStar:
    def __init__(self, graph, map_letter_point):
        self.graph = graph
        self.map_letter_point = map_letter_point

    def __RevertPath__(self, path):
        list = []
        actualPosition = 'target'
        while actualPosition != 'initial':
            list.append(actualPosition)
            actualPosition = path[actualPosition]
        list.append("initial")
        list.reverse()

        return list

    def __EuclidianDistance__(self, src, dst):
        x1 = self.map_letter_point[src].x
        y1 = self.map_letter_point[src].y
        x2 = self.map_letter_point[dst].x
        y2 = self.map_letter_point[dst].y

        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def run(self):

        visitations = 0
        path = {}
        dst = {}
        findSolution = False
        threshold = self.__EuclidianDistance__('initial', 'target')
        dst['initial'] = 0

        while not findSolution:

            stack = [('initial', 0)]
            thresholdList = []
            while len(stack) > 0:
                topOfStack = stack.pop()
                actualPosition = topOfStack[0]
                Gx = topOfStack[1]
                Hx = self.__EuclidianDistance__(actualPosition, 'target')
                Fx = Gx + Hx

                if Fx > threshold:
                    thresholdList.append(Fx)
                    continue

                visitations += 1
                
                if actualPosition == 'target':
                    findSolution = True
                    break

                for child in self.graph[actualPosition]:
                    stack.append((child, Fx))

                    if child not in path:
                        path[child] = actualPosition
                        dst[child] = dst[actualPosition] + self.__EuclidianDistance__(actualPosition, child)

            threshold = min(thresholdList)

        path = self.__RevertPath__(path)
        return (visitations, dst['target'], path)