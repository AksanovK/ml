import pygame
import numpy as np
from sklearn.svm import SVC


def draw_dot(points, event, current_class, screen):
    color = ORANGE
    if current_class == 1:
        color = BLUE
    pygame.draw.circle(screen, color, event.pos, radius)
    points.append([[event.pos[0], event.pos[1]], current_class, None])


def get_line(svc):
    w = svc.coef_[0]
    a = -w[0] / w[1]
    xx = np.array([0, width])
    yy = a * xx - (svc.intercept_[0]) / w[1]
    return np.array(list(zip(xx, yy)))


def draw_pygame():
    screen = pygame.display.set_mode((width, height))
    screen.fill(WHITE)
    coordinates = []
    line_printed = False
    action = True
    while action:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                action = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and line_printed is False:
                    draw_dot(coordinates, event, 0, screen)
                if event.button == 3 and line_printed is False:
                    draw_dot(coordinates, event, 1, screen)
                if line_printed is True:
                    pos = list(event.pos)
                    cls = svc.predict([pos])[0]
                    points.append([pos, cls, None])
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and line_printed is False:
                line_printed = True
                x_coordinates = np.array(list(map(lambda p: p[0], coordinates)))
                y_coordinates = np.array(list(map(lambda p: p[1], coordinates)))
                svc = SVC(kernel='linear')
                svc.fit(x_coordinates, y_coordinates)
                line = get_line(svc)
                p1 = line[0]
                p2 = line[-1]
                pygame.draw.line(screen, 'black', p1, p2, 2)
        count_sim = 0
        count_diff = 0
        for point in points:
            is_above = lambda point, p1, p2: np.cross(point - p1, p2 - p1) < 0
            if is_above(point[0], p1, p2):
                point[2] = 0
            else:
                point[2] = 1
            if point[1] == point[2]:
                count_sim = count_sim + 1
            else:
                count_diff = count_diff + 1
            if count_sim > count_diff:
                pygame.draw.circle(screen, colors[point[2]], point[0], radius)
            else:
                pygame.draw.circle(screen, colors[point[1]], point[0], radius)

        pygame.display.update()


if __name__ == '__main__':
    width, height = 800, 500
    WHITE = (255, 255, 255)
    BLUE = (25, 25, 112)
    ORANGE = (255, 127, 80)
    radius = 8
    points = []
    colors = {0: ORANGE, 1: BLUE}
    draw_pygame()
