import time
import numpy as np
import math
import pygame


def dist(p_i, p_j):
    return np.sqrt(np.sum((np.array(p_i) - np.array(p_j)) ** 2))


def dbscan_1(points):
    points_green = []
    points_without_green = points.copy()

    for pnt in points:
        neighbor = 0
        for point in points:
            if dist(pnt[:-1], point[:-1]) < eps and pnt != point:
                neighbor += 1

        if neighbor >= minNeighbor:
            point_green = ([pnt[:-1], "green"])
            points_green.append(point_green)
            points_without_green.remove(pnt)

    draw(points_green)
    search_yellow(points_green, points_without_green)


def search_yellow(points_green, points_without_green):
    points_yellow = []
    for pnt in points_without_green:
        neighbor = 0
        for point in points_green:
            if dist(pnt[:-1], point[:-1]) < eps:
                neighbor += 1
        if 1 <= neighbor < minNeighbor:
            point_yellow = ([pnt[:-1], "yellow"])
            points_yellow.append(point_yellow)

    draw(points_yellow)


def draw(points):
    for point in points:
        if point[-1] == "green":
            # print(point[0])
            pygame.draw.circle(screen, 'green', point[0], R)
        if point[-1] == "yellow":
            pygame.draw.circle(screen, 'yellow', point[0], R)
        if point[-1] == "red":
            pygame.draw.circle(screen, 'red', point[0], R)

    pygame.display.update()


minNeighbor = 3
eps = 20
R = 5
screen = pygame.display.set_mode([800, 600])

UNCLASSIFIED = False
NOISE = None


def _dist(p, q):
    return math.sqrt(np.power(p - q, 2).sum())


def _eps_neighborhood(p, q, eps):
    return _dist(p, q) < eps


def _region_query(m, point_id, eps):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        if _eps_neighborhood(m[:, point_id], m[:, i], eps):
            seeds.append(i)
    return seeds


def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id

        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                            classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True


def dbscan(m, eps, min_points):
    cluster_id = 1
    n_points = m.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications


def pygame_func():
    pygame.init()
    points_x = []
    points_y = []
    points = []
    points_1 = []

    screen = pygame.display.set_mode([600, 600])
    screen.fill(color='white')
    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return points, classification

            if pygame.mouse.get_pressed()[0]:
                pygame.draw.circle(screen, 'black', event.pos, 5)
                x, y = event.pos
                points_x.append(x)
                points_y.append(y)
                coord = ([*event.pos, -1])
                points_1.append(coord)
                time.sleep(0.05)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    points.insert(0, points_x)
                    points.insert(1, points_y)
                    points = np.array(points)
                    eps = 20
                    min_points = 3
                    classification = dbscan(points, eps, min_points)

                    for i in range(len(classification)):
                        if classification[i] is None:
                            classification[i] = 0

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    dbscan_1(points_1)

        pygame.display.update()


def draw_clustering(point_x, point_y, color):
    pygame.draw.circle(screen, color, (point_x, point_y), 5)
    pygame.display.update()


if __name__ == '__main__':
    colors = ["red", "green", "yellow", "blue", "#FF00FF", "#00FF00", "#00FFFF", "#800080", "#8B4513", "#FF69B4"]
    points, classification = pygame_func()

    screen = pygame.display.set_mode([600, 600])
    screen.fill(color='white')
    pygame.display.update()

    for i in range(len(classification)):
        draw_clustering(points[0][i], points[1][i], colors[classification[i]])

