from enum import Enum
from dataclasses import dataclass


class NODE_TYPES(Enum):
    depot = 0
    customer = 1
    charging_station = 2

@dataclass
class Node:
    x: int
    y: int
    node_type: NODE_TYPES
    visited_order: int
