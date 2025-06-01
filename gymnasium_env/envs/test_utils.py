import unittest
from utils import generate_packages_weight, calc_energy_consumption
from utils import total_distance_of_a_random_route
import gymnasium
import gymnasium_env


class test_utils(unittest.TestCase):
    def setUp(self):
        super().setUp()
        env = gymnasium.make(
            id="gymnasium_env/DroneTsp-v1",
            render_mode='human',
            num_customer_nodes=5,
            num_charge_nodes=1,
            package_weight=40,
            max_energy=-1
        )
        observation, info = env.reset()
        self.customers = info['customers']

    def test_generate_packages_weight(self):
        """Kiểm tra hàm generate_packages_weight
        """

        result = generate_packages_weight(max_weight=0, total_packages=10)
        self.assertEqual(first=len(result) , second=0, msg="Max weight is 0 but result isn't an empty array.")
        result = generate_packages_weight(max_weight=40, total_packages=0)
        self.assertEqual(first=len(result), second=0, msg="Total packages is 0 but result isn't an empty array.")

        with self.assertRaises(expected_exception=ValueError) as cm:
            generate_packages_weight(-10, 10)
        self.assertIn("Max weight and total packages can't be negative.", str(cm.exception))
        with self.assertRaises(expected_exception=ValueError):
            generate_packages_weight(40, -10)
        self.assertIn("Max weight and total packages can't be negative.", str(cm.exception))

        result = generate_packages_weight(max_weight=40, total_packages=10)
        self.assertEqual(first=len(result), second=10, msg="Package quantity is wrong.")
        self.assertEqual(first=sum(result), second=40, msg="Total weight is wrong.")

    def test_calc_energy_consumption(self):
        """Kiểm tra hàm calc_energy_consumption
        """

        with self.assertRaises(expected_exception=ValueError) as cm:
            calc_energy_consumption(gij=-10)
        self.assertIn("Weight can't be negative.", str(cm.exception))

        energy_consumption = calc_energy_consumption(gij=10)
        self.assertEqual(first=energy_consumption, second=22753.08, msg="Result is different with the expected value.")

    def test_total_distance_of_a_random_route(self):
        """Kiểm tra hàm total_distance_of_a_random_route
        """
        result = total_distance_of_a_random_route(nodes=self.customers)
        self.assertGreater(a=result, b=0, msg="Distance can't be negative.")

if __name__ == '__main__':
    unittest.main()
