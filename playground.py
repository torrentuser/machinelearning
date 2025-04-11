import random
from math import pi

from geopy.distance import distance
from geopy.point import Point

l = 86.96
r = 36.5
c = pi * r
d = 400


def rad2ang(rad):
    return (rad / pi) * 180


class Playground:
    def __init__(self, point, direction):
        self.start = point
        self.direction = direction
        self.center_1 = distance(meters=r).destination(self.start, direction + 90)
        self.center_2 = distance(meters=l).destination(self.center_1, direction - 180)
        self.anchor_1 = distance(meters=r).destination(self.center_1, direction + 90)
        self.anchor_2 = distance(meters=l).destination(self.start, direction + 180)

    def coordinate(self, x):
        x = x % d

        if x < c:
            angle = self.direction - 90 + rad2ang(x / r)
            return distance(meters=r).destination(self.center_1, angle)
        x = x - c
        if x < l:
            angle = self.direction + 180
            return distance(meters=x).destination(self.anchor_1, angle)
        x = x - l
        if x < c:
            angle = self.direction + 90 + rad2ang(x / r)
            return distance(meters=r).destination(self.center_2, angle)
        x = x - c
        return distance(meters=x).destination(self.anchor_2, self.direction)

    def random_offset(self, x):
        angle = random.randint(0, 360)
        offset = random.uniform(0, 0.5)
        coord = self.coordinate(x)
        return distance(meters=offset).destination(coord, angle)


playgrounds = {
    38: Playground(Point(31.291805, 121.502805), 180),  # 邯郸南区田径场夜跑
    33: Playground(Point(31.291805, 121.502805), 180),  # 邯郸南区田径场课外活动
    28: Playground(Point(31.291805, 121.502805), 180),  # 邯郸南区田径场早操
    34: Playground(Point(31.296757, 121.507009), 216.5),  # 菜地课外
    39: Playground(Point(31.296757, 121.507009), 216.5),  # 菜地夜跑
    35: Playground(Point(31.335097, 121.502149), 166.3),  # 江湾课外
    40: Playground(Point(31.335097, 121.502149), 166.3),  # 江湾夜跑
    30: Playground(Point(31.335097, 121.502149), 166.3)  # 江湾早锻炼
}
