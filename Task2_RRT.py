import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

# генерация препятствий
def setObstacles(N, size=1, mu=0.1, sigma=0.1):
    points, rad = size * np.random.random_sample((N, 2)), np.random.normal(mu, sigma, N)
    # от цетнра препятствия до начала
    l = np.power(np.sum(np.power(points, 2), axis=1), 0.5)
    # до конца
    w = np.power(np.sum(np.power(size - points, 2), axis=1), 0.5)
    indexes = tuple(np.where(np.logical_and(np.logical_and(l > rad, w > rad), rad > 0)))
    return np.column_stack((points[indexes], rad[indexes]))  #  FIX IT!

# эвристическая функция для точки
def evristika(x, y, size=1):
    return math.hypot(size, size) - math.hypot((size - x), (size - y))

# пересечение ребра с препятствием
def line_circle_intersection(edge, obstacles):
    return line_circle_intersection(edge[0], edge[1], obstacles)

# проверка на прохождение через препятствие ребра между двумя вершинами
def line_circle_intersection(qnearest, qnew, obstacles):
    # K и B - коэфициенты в уравнении прямой y = Kx + B
    # center - координаты центра
    intersection_coord = []
    K = (qnew[1] - qnearest[1]) / (qnew[0] - qnearest[0])
    B = (qnew[0] * qnearest[1] - qnearest[0] * qnew[1]) / (qnew[0] - qnearest[0])
    for ob in obstacles:
        a = 1 + K ** 2
        b = -2 * ob[0] + 2 * K * B - 2 * K * ob[1]
        c = -ob[2] ** 2 + (B - ob[1]) ** 2 + ob[0] ** 2
        D = b ** 2 - 4 * a * c
        # scipy find D
        if D > 0:
            x1 = (-b - math.sqrt(D)) / (2 * a)
            x2 = (-b + math.sqrt(D)) / (2 * a)
            if (qnew[0] < x1 and x1 < qnearest[0]) or (qnearest[0] < x1 and x1 < qnew[0]):
                intersection1 = (x1, K * x1 + B)
                intersection_coord.append(intersection1)
            if (qnew[0] < x2 and x2 < qnearest[0]) or (qnearest[0] < x2 and x2 < qnew[0]):
                intersection2 = (x2, K * x2 + B)
                intersection_coord.append(intersection2)
            if len(intersection_coord) != 0:
                return intersection_coord
    return intersection_coord

#ищет лучшую ноду в графе и возвращает её
def findBestNode(G, probability = 0.8):
    if np.random.random_sample() > probability:
        # выдать случайную вершину
        random_number = np.random.randint(0, G.number_of_nodes())
        return (list(G.nodes)[random_number])
    else:
        # выдать вершину с максимальной эвристикой
        nodesAsList = list(G.nodes)
        nodesAsArr = np.asarray(nodesAsList)
        maxs = np.max(nodesAsArr, axis=0)[2]
        row = np.where(nodesAsArr == maxs)[0][0]
        return nodesAsList[row]

# возвращает случайную точку в области ноды
def randomNode(Node, obstacles, size=1, allowInObstacle=False, circle_r = 1):
    while True:
        # random angle and radius
        r, angle = circle_r * math.sqrt(np.random.random()), 2 * math.pi * np.random.random_sample()
        # calculating coordinates and evristics
        x, y = r * math.cos(angle) + Node[0], r * math.sin(angle) + Node[1]
        e = evristika(x, y, size)
        if 0 <= x <= size and 0 <= y <= size:
            if not allowInObstacle and PointInObstcle((x, y, e), obstacles):
                continue
            else:
                break
    return (x, y, e)

# отрисовка всего окружения
def drawEnv(G, start, finish, size):
    fig, ax = plt.subplots()
    for point in obstacles:
        circle = plt.Circle((point[0], point[1]), point[2], color='r')
        ax.add_artist(circle)

    for point in nx.nodes(G):
        plt.plot(point[0], point[1], 'b.')

    for edge in nx.edges(G):
        p1 = [edge[0][0], edge[1][0]]
        p2 = [edge[0][1], edge[1][1]]
        plt.plot(p1, p2, 'k-')

    # вывод пути
    if G.has_node(finish):
        previous_node = start
        length = 0
        for point in nx.shortest_path(G, start, finish):
            plt.plot(point[0], point[1], 'go')
            if not previous_node == point:
                p1 = [previous_node[0], point[0]]
                p2 = [previous_node[1], point[1]]
                length += math.hypot(point[0] - previous_node[0], point[1] - previous_node[1])
                previous_node = point
                plt.plot(p1, p2, 'g-')
        print('Length = ', length)
    plt.axis([0, size, 0, size])
    plt.show()

# проверка, не лежит ли точка внутри препятствия
def PointInObstcle(qnew, obstacles):
    # попробовать использовать numpy : all, where
    for obj in obstacles:
        if math.hypot(qnew[0] - obj[0], qnew[1] - obj[1]) < obj[2]:
            return True
    return False

# поиск ближ вершины
def findNearestNode(G, qnew, obstacles, start, size=1, allowEdge=True):
    min_length = math.hypot(size, size)
    list_nearest_node = [min_length, start]
    for node in nx.nodes(G):
        length = math.hypot(node[0] - qnew[0], node[1] - qnew[1])
        if length < min_length:
            if len(line_circle_intersection(qnew, node, obstacles)) == 0:
                min_length = length
                list_nearest_node = [min_length, node]
    if allowEdge:
        for edge in G.edges():
            nodeOnEdge = findNodeOnEdge(edge, qnew)
            if len(nodeOnEdge) != 0: #True, если смогли построить перпендикуляр
                if len(line_circle_intersection(qnew, nodeOnEdge, obstacles)) == 0:  #True, если нет препятствий на пути
                    length = math.hypot(nodeOnEdge[0] - qnew[0], nodeOnEdge[1] - qnew[1])
                    if length < min_length:

                        min_length = length
                        list_nearest_node = [min_length, nodeOnEdge, edge[0], edge[1]]
    if len(list_nearest_node) == 4:
        # nearest edge
        G.add_node(qnew)
        G.add_node(list_nearest_node[1])     # добавление новой вершины, лежащей на ребре
        G.remove_edge(list_nearest_node[2], list_nearest_node[3]) #удалить ребро
        G.add_edge(list_nearest_node[2], list_nearest_node[1]) #добавление полуребра
        G.add_edge(list_nearest_node[3], list_nearest_node[1]) #добавление полуребра
        G.add_edge(list_nearest_node[1], qnew)    #добавление связи между точкой и ребром
    if len(list_nearest_node) == 2 and len(line_circle_intersection(qnew, list_nearest_node[1], obstacles)) == 0:
        G.add_node(qnew)
        G.add_edge(list_nearest_node[1], qnew)
    return G

# строит перпендикуляр от точки до прямой, если может - возвращает координаты, если нет - пустой список
def findNodeOnEdge(edge, point):
    point1, point2 = edge[0], edge[1]
    L = math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2)
    PR = (point[0] - point1[0]) * (point2[0] - point1[0]) + (point[1] - point1[1]) * (point2[1] - point1[1])
    res = True
    cf = PR / L
    if cf < 0:
        cf=0
        res=False
    if cf > 1:
        cf=1
        res=False
    xres = point1[0] + cf * (point2[0] - point1[0])
    yres = point1[1] + cf * (point2[1] - point1[1])
    evristic = evristika(xres, yres)
    if res:
        return (xres, yres, evristic)
    return tuple()

if __name__ == '__main__':
    # Некоторые параметры
    K = 500
    # размер, старт/финиш
    size = 1
    start = (0, 0, evristika(0, 0))
    finish = (size, size, evristika(size, size))

    print('Введите количество препятствий')
    N = int(input())
    while N < 0:
        print('Неправильное число, введите новое число:')
        N = int(input())

    # генерация препятствий
    obstacles = setObstacles(N)

    # добавление точки старта
    G = nx.Graph()
    G.add_node(start)

    # новый коммент
    # Проверка на прохождение по прямой
    if len(line_circle_intersection((start), (finish), obstacles)) == 0:
        G.add_node(finish)
        G.add_edge(start, finish)
    else:
        for _ in range(K):
            # поиск вершины с наилучшей эвристикой и построение точки в области этой вершины
            qnew = randomNode(findBestNode(G), obstacles, size)
            # поиск ближайшей вершины или ребра и создаение связи
            G = findNearestNode(G, qnew, obstacles, start, size)
            # если есть возможность пройти по прямой от точки к финишу, всё кульно
            # drawEnv(G, start, finish, size)
            if G.has_node(qnew) and len(line_circle_intersection(qnew, finish, obstacles)) == 0:
                G.add_node(finish)
                G.add_edge(qnew, finish)
                break
        # отрисовка среды
    drawEnv(G, start, finish, size)