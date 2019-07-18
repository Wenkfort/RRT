import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

# генерация препятствий
def setObstacles(N, size=1, mu=0.1, sigma=0.1):
    points = size * np.random.random_sample((N, 2))
    rad = np.random.normal(mu, sigma, N)

    # от цетнра препятствия до начала
    l = np.power(np.sum(np.power(points, 2), axis=1), 0.5)
    # до конца
    w = np.power(np.sum(np.power(size - points, 2), axis=1), 0.5)

    indexes = tuple(np.where(np.logical_and(np.logical_and(l > rad, w > rad), rad > 0)))

    return np.column_stack((points[indexes], rad[indexes]))  #  FIX IT!

#возвращает ближайшую ноду
def nearestNode(G, qnew, obstacles, edge = False):
    min = np.sqrt(2)
    qnearest = (0, 0, 0)
    # поиск ближайшей вершины
    for g in G:
        length = math.hypot(g[0] - qnew[0], g[1] - qnew[1])
        if length < min:
            min = length
            qnearest = g
    # поиск ближайшего ребра
    if edge:
        for ed in G.edges():
            g = findNodeOnEdge(ed[0], ed[1], qnew)
            if len(g) != 0:
                if len(line_circle_intersection(qnew, g, obstacles)) == 0:
                    length = math.hypot(g[0] - qnew[0], g[1] - qnew[1])
                    if length < min:
                        min = length
                        qnearest = g

    return qnearest

# строит перпендикуляр от точки до прямой, если может - возвращает координаты, если нет - пустой список
def findNodeOnEdge(point1, point2, point):
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
    evristic = math.sqrt(2) - math.hypot((1 - xres), (1 - yres))
    if res:
        return (xres, yres, evristic)
    return tuple()

# отрисовка всего окружения
def drawEnv(G):
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

    plt.axis([0, 1, 0, 1])
    plt.show()

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
            if (x1 < qnearest[0] and x1 > qnew[0]) or (x1 > qnearest[0] and x1 < qnew[0]):
                intersection1 = (x1, K * x1 + B)
                intersection_coord.append(intersection1)
            if (x2 < qnearest[0] and x2 > qnew[0]) or (x2 > qnearest[0] and x2 < qnew[0]):
                intersection2 = (x2, K * x2 + B)
                intersection_coord.append(intersection2)
            if len(intersection_coord) != 0:
                return intersection_coord
    return intersection_coord

# проверка, не лежит ли точка внутри препятствия
def PointInObstcle(qnew, obstacles):
    # попробовать использовать numpy : all, where
    for obj in obstacles:
        if math.hypot(qnew[0] - obj[0], qnew[1] - obj[1]) < obj[2]:
            return True
    return False

#ищет лучшую ноду в графе
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
def randomNode(Node, obstacles, circle_r = 1, size = 1):
    # random angle and radius
    r, angle = circle_r * math.sqrt(np.random.random()), 2 * math.pi * np.random.random_sample()
    # calculating coordinates and evristics
    x, y = r * math.cos(angle) + Node[0], r * math.sin(angle) + Node[1]
    e = math.sqrt(2) - math.hypot((1 - x), (1 - y))

    while (x < 0 or x > size) or (y < 0 or y > size):
        # random angle
        r, angle = circle_r * math.sqrt(np.random.random()), 2 * math.pi * np.random.random_sample()
        # calculating coordinates and evristics
        x, y = r * math.cos(angle) + Node[0], r * math.sin(angle) + Node[1]
        e = math.sqrt(2) - math.hypot((1 - x), (1 - y))
    return (x, y, e)


if __name__ == '__main__':
    #ввод
    print('Введите количество препятствий')
    N = int(input())
    while N < 0:
        print('Неправильное число, введите новое число:')
        N = int(input())

    # генерация N препятствий
    obstacles = setObstacles(N)

    # создание дерева
    start = (0, 0, 0)  # something hashable, format: (x, y, f(e)) f(e) - эвристическая функция
    finish = (1, 1, np.sqrt(2)) # something hashable, format: (x, y, f(e))
    G = nx.Graph()
    G.add_node(start)

    K = 500   #количество иттераций

    if len(line_circle_intersection(start, finish, obstacles)) == 0:
        G.add_node(finish)
        G.add_edge(start, finish)
    else:
        for k in range(K):
            #выбор точки в дереве с наибольшей эвристикой, генерация рядом с ней случайной точки
            qnew = randomNode(findBestNode(G), obstacles)
            # поиск ближайшей вершины в графе G
            qnearest = nearestNode(G, qnew, obstacles)
            # проверка пересечения прямой и препятствий при движении qnearest -> qnew
            if len(line_circle_intersection(qnearest, qnew, obstacles)) == 0:
                G.add_node(qnew)
                G.add_edge(qnearest, qnew)
                # если есть возможность пройти по прямой от точки к финишу, всё кульно
                if len(line_circle_intersection(qnew, finish, obstacles)) == 0:
                    G.add_node(finish)
                    G.add_edge(qnew, finish)
                    print('количество итераций:', k+1)
                    break
    # отрисовка среды
    drawEnv(G)