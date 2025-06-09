# file: node_transformer.py (hoặc giữ node_encoder.py nếu bạn không thích rename)
"""
Cung cấp các hàm chuyển đổi giữa đối tượng Node và mảng numpy để phục vụ cho việc encode/decode trong môi trường Drone TSP.
"""
import numpy as np
from gymnasium_env.envs.interfaces import Node, NODE_TYPES

class NodeTransformer:
    """
    Lớp tiện ích để mã hóa (encode) và giải mã (decode) đối tượng Node thành mảng numpy và ngược lại.
    Dùng cho việc xử lý dữ liệu trong môi trường học tăng cường.
    """
    STRUCT = ["lon", "lat", "node_type", "package_weight", "visited_order", "start_time", "end_time", "visited_time"]

    @staticmethod
    def encode(node: Node) -> np.ndarray:
        """
        Mã hóa một đối tượng Node thành mảng numpy.
        Args:
            node (Node): Đối tượng Node cần mã hóa.
        Returns:
            np.ndarray: Mảng numpy biểu diễn các thuộc tính của node.
        Raises:
            TypeError: Nếu đầu vào không phải là Node.
        """
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
        """
        Giải mã một mảng numpy thành đối tượng Node.
        Args:
            arr (np.ndarray): Mảng numpy hoặc list chứa thông tin node.
        Returns:
            Node: Đối tượng Node được khôi phục từ mảng.
        Raises:
            TypeError: Nếu đầu vào không phải array-like.
            ValueError: Nếu số lượng phần tử không đúng.
        """
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
        """
        Trả về số lượng thuộc tính của một node (dùng cho shape của observation space).
        Returns:
            int: Số chiều của vector node.
        """
        return len(NodeTransformer.STRUCT)
