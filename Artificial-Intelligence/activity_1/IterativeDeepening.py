class ID:
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

        return ((x2-x1)**2 + (y2-y1)**2)**0.5

    def run(self):

        visitations = 0
        maxDepth = 0
        path = {}
        dst = {}
        findSolution = False
        dst['initial'] = 0

        while not findSolution:

            maxDepth += 1
            stack = [('initial', 0)]
            while len(stack) > 0:
                topOfStack = stack.pop()
                depth = topOfStack[1]
                actualPosition = topOfStack[0]

                if depth > maxDepth:
                    continue
                visitations += 1
                if actualPosition == 'target':
                    findSolution = True
                    break

                for child in self.graph[actualPosition]:
                    stack.append((child, depth + 1))

                    if child not in path:
                        path[child] = actualPosition
                        dst[child] = dst[actualPosition] + self.__EuclidianDistance__(actualPosition, child)

        path = self.__RevertPath__(path)
        return (visitations, dst['target'], path)
