import numpy as np
from gymnasium_env.envs.interfaces import Node, NODE_TYPES


class NodeEncoder:
    STRUCT = ["x", "y", "node_type", "package_weight", "visited_order"]

    @staticmethod
    def encode(node: Node) -> np.ndarray:
        if not isinstance(node, Node):
            raise TypeError(f"Expected Node, got {type(node)}")

        # validate all fields
        if not isinstance(node.x, int): raise TypeError("x must be int")
        if not isinstance(node.y, int): raise TypeError("y must be int")
        if not isinstance(node.node_type, NODE_TYPES): raise TypeError("node_type must be NODE_TYPES")
        if not isinstance(node.package_weight, float): raise TypeError("package_weight must be float")
        if not isinstance(node.visited_order, int): raise TypeError("visited_order must be int")

        return np.array([
            node.x,
            node.y,
            node.node_type.value,
            node.package_weight,
            node.visited_order
        ], dtype=np.float32)

    @staticmethod
    def get_shape() -> int:
        return 5  # length of STRUCT
