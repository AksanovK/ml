import sys
from math import hypot
import random as rnd
import pygame


def generate_points(mean_x, mean_y, deviation_x, deviation_y):
    return rnd.gauss(mean_x, deviation_x), rnd.gauss(mean_y, deviation_y)


def dbscan_naive(P, eps, m, distance):
    NOISE = 0
    C = 0

    visited_points = set()
    clustered_points = set()
    clusters = {NOISE: []}

    def region_query(p):
        return [q for q in P if distance(p, q) < eps]

    def expand_cluster(p, neighbours):
        if C not in clusters:
            clusters[C] = []
        clusters[C].append(p)
        clustered_points.add(p)
        while neighbours:
            q = neighbours.pop()
            if q not in visited_points:
                visited_points.add(q)
                neighbourz = region_query(q)
                if len(neighbourz) > m:
                    neighbours.extend(neighbourz)
            if q not in clustered_points:
                clustered_points.add(q)
                clusters[C].append(q)
                if q in clusters[NOISE]:
                    clusters[NOISE].remove(q)

    for p in P:
        if p in visited_points:
            continue
        visited_points.add(p)
        neighbours = region_query(p)
        if len(neighbours) < m:
            clusters[NOISE].append(p)
        else:
            C += 1
            expand_cluster(p, neighbours)

    return clusters


if __name__ == '__main__':
    eps = 100
    min = 3
    width, height = 800, 500
    WHITE = (255, 255, 255)
    color = [
        (255, 128, 0),
        (255, 0, 0),
        (255, 255, 0),
        (128, 255, 0),
        (0, 255, 255),
        (0, 255, 0),
        (0, 0, 255),
        (255, 51, 255),
        (255, 51, 153),
        (0, 0, 0),
    ]
    pygame.init()
    screen = pygame.display.set_mode((800, 500))
    r = pygame.Rect(0, 0, 1200, 700)
    pygame.draw.rect(screen, (255, 255, 255), r, 0)
    pygame.draw.polygon(screen, (255, 0, 0), ((0, 0), (0, 700)), width=1)
    pygame.draw.polygon(screen, (255, 0, 0), ((0, 700), (1200, 700)), width=1)

    # screen = pygame.display.set_mode((width, height))
    # screen.fill(WHITE)

    points = set()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                pygame.draw.rect(screen, (255, 255, 255), r, 0)
                pygame.draw.polygon(screen, (255, 0, 0), ((0, 0), (0, 700)), width=10)
                pygame.draw.polygon(screen, (255, 0, 0), ((0, 700), (1200, 700)), width=10)
                clusters = dbscan_naive(list(points), eps, min, lambda x, y: hypot(x[0] - y[0], x[1] - y[1]))
                points.clear()
                print(clusters)
                for cluster in clusters.keys():
                    for point in clusters.get(cluster):
                        pygame.draw.circle(screen, color[cluster], point, 5, width=0)
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                if pos not in points:
                    points.add(pos)
                    pygame.draw.circle(screen, (0, 0, 0), pos, 5, width=0)
        pygame.display.flip()




