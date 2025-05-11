import numpy as np
from gymnasium_env.envs.interfaces import Node, NODE_TYPES


class NodeEncoder:
    STRUCT = ["lon", "lat", "node_type", "package_weight", "visited_order", "start_time", "end_time"]

    @staticmethod
    def encode(node: Node) -> np.ndarray:
        if not isinstance(node, Node):
            raise TypeError(f"Expected Node, got {type(node)}")

        # validate all fields
        if not isinstance(node.lon, float): raise TypeError("lon must be float")
        if not isinstance(node.lat, float): raise TypeError("lat must be float")
        if not isinstance(node.node_type, NODE_TYPES): raise TypeError("node_type must be NODE_TYPES")
        if not isinstance(node.package_weight, float): raise TypeError("package_weight must be float")
        if not isinstance(node.visited_order, int): raise TypeError("visited_order must be int")
        if not isinstance(node.time_window, tuple): raise TypeError("time_window must be tuple")

        return np.array([
            node.lon,
            node.lat,
            node.node_type.value,
            node.package_weight,
            node.visited_order,
            node.time_window[0], # start time
            node.time_window[1] # end time
        ], dtype=np.float32)

    @staticmethod
    def get_shape() -> int:
        return len(NodeEncoder.STRUCT)
