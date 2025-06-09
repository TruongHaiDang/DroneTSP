"""
Định nghĩa các kiểu node và cấu trúc dữ liệu node sử dụng trong môi trường Drone TSP.
"""
from enum import Enum
from dataclasses import dataclass


class NODE_TYPES(Enum):
    """
    Enum biểu diễn các loại node trong bài toán Drone TSP:
        - depot: điểm xuất phát/kết thúc
        - customer: khách hàng nhận hàng
        - charging_station: trạm sạc
    """
    depot = 0
    customer = 1
    charging_station = 2

@dataclass
class Node:
    """
    Cấu trúc dữ liệu cho một node trong môi trường Drone TSP.
    Bao gồm thông tin vị trí, loại node, trọng lượng gói hàng, thứ tự ghé thăm và các mốc thời gian.
    """
    lon: float
    lat: float
    node_type: NODE_TYPES
    package_weight: float
    visited_order: int
    start_time: float
    end_time: float
    visited_time: float
