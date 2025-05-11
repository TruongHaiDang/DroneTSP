from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple


class NODE_TYPES(Enum):
    depot = 0
    customer = 1
    charging_station = 2

@dataclass
class Node:
    lon: float
    lat: float
    node_type: NODE_TYPES
    package_weight: float
    visited_order: int
    time_window: Optional[Tuple[float, float]] = None
