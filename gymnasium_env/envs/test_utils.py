import unittest
from interfaces import Node, NODE_TYPES
from utils import euclidean_distance, generate_packages_weight, calc_energy_consumption


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_euclidean_distance(self):
        x1, y1 = 1, 2
        x2, y2 = 4, 6
        node_1 = Node(x=x1, y=y1, node_type=NODE_TYPES.customer, package_weight=0, visited_order=1)
        distance_1 = euclidean_distance(node_1=node_1, node_2=node_1)
        self.assertEqual(first=distance_1, second=0, msg="Cùng một node nhưng khoảng cách khác 0.")
        node_2 = Node(x=x2, y=y2, node_type=NODE_TYPES.customer, package_weight=0, visited_order=1)
        distance_2 = euclidean_distance(node_1=node_1, node_2=node_2)
        self.assertEqual(first=distance_2, second=5, msg="Kết quả không giống kết quả đã biết trước.")

    def test_generate_packages_weight(self):
        total_packages = 10
        total_weight = 40
        packages = generate_packages_weight(max_weight=total_weight, total_packages=total_packages)
        self.assertEqual(first=len(packages), second=total_packages, msg="Không đủ số lượng hàng hoá.")
        self.assertEqual(first=sum(packages), second=total_weight, msg="Không đủ khối lượng hàng hoá.")

    def test_energy_consumption(self):
        gij_1 = 0
        gij_2 = 15
        energy_1 = calc_energy_consumption(gij=gij_1)
        energy_2 = calc_energy_consumption(gij=gij_2)
        self.assertGreater(a=energy_2, b=energy_1, msg="Mang theo khối lượng hàng lớn hơn nhưng năng lượng tiêu thụ nhỏ hơn.")
        with self.assertRaises(ValueError, msg="Không raise ValueError khi nhập khối lượng âm."):
            calc_energy_consumption(gij=-10)

if __name__ == '__main__':
    unittest.main()
