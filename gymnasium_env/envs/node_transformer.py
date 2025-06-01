# file: node_transformer.py (hoặc giữ node_encoder.py nếu bạn không thích rename)
import numpy as np
from gymnasium_env.envs.interfaces import Node, NODE_TYPES

class NodeTransformer:
    STRUCT = ["lon", "lat", "node_type", "package_weight", "visited_order", "start_time", "end_time", "visited_time"]

    @staticmethod
    def encode(node: Node) -> np.ndarray:
        if not isinstance(node, Node):
            raise TypeError(f"Expected Node, got {type(node)}")

        return np.array([
            node.lon,
            node.lat,
            node.node_type.value,
            node.package_weight,
            node.visited_order,
            node.start_time,
            node.end_time,
            node.visited_time
        ], dtype=np.float32)

    @staticmethod
    def decode(arr: np.ndarray) -> Node:
        if not isinstance(arr, (list, tuple, np.ndarray)):
            raise TypeError("Expected array-like input")
        if len(arr) != len(NodeTransformer.STRUCT):
            raise ValueError(f"Expected {len(NodeTransformer.STRUCT)} elements, got {len(arr)}")

        return Node(
            lon=float(arr[0]),
            lat=float(arr[1]),
            node_type=NODE_TYPES(int(arr[2])),
            package_weight=float(arr[3]),
            visited_order=int(arr[4]),
            start_time=float(arr[5]),
            end_time=float(arr[6]),
            visited_time=float(arr[7])
        )

    @staticmethod
    def get_shape() -> int:
        return len(NodeTransformer.STRUCT)
