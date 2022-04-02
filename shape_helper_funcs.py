import numpy as np
from shapely.geometry import Point, Polygon


def rotate(v, alpha):
    """
    rotate 2d np vector with rotation matrix

    :param v: 2d vector
    :param alpha: radians
    :return: rotated vector
    """
    matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    return v @ matrix


def normalize(v):
    """
    scale vector to unit length

    :param v: np vector
    :return: scaled np vector
    """
    d = np.linalg.norm(v, 2)
    return v / d


def get_polygons(v1, v_next, num, k=500):
    """
    gets shapely polygons from direction calculated from v1 and v_next

    :param v1: current position
    :param v_next: next position
    :param num: number of polygons
    :param k: size of polygons
    :return: list of polygons
    """
    polygons = []
    for i in range(0, num*2, 2):
        alpha = np.pi/num
        v = v_next - v1
        v = normalize(v)
        v = rotate(v, alpha*i)
        p1 = Point(v1)
        p2 = Point(v1 + rotate(v, alpha)*k)
        p3 = Point(v1 + rotate(v, -alpha)*k)
        poly = Polygon([p1, p2, p3])
        polygons.append(poly)
    return polygons
