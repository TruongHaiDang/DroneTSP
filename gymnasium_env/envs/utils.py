import numpy as np
from interfaces import Node
import random


def euclidean_distance(node_1: Node, node_2: Node):
    x1, y1 = node_1.x, node_1.y
    x2, y2 = node_2.x, node_2.y
    return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

def generate_package_weight(max_weight: float, total_package: int):
    result = []
    
    remain_weight = max_weight
    for i in range(total_package):
        if i < total_package:
            new_package_weight = random.randint(0, remain_weight)
            result.append(new_package_weight)
            remain_weight -= new_package_weight
        else:
            result.append(remain_weight)
            remain_weight = 0

    return result

def energy_consumption(gij: float):
    pass
