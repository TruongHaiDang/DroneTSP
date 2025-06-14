import unittest
import gymnasium
import gymnasium_env

class TestDroneChargeLimit(unittest.TestCase):
    def test_truncated_when_exceed_charge_limit(self):
        env = gymnasium.make(
            id="gymnasium_env/DroneTsp-v1",
            render_mode=None,
            num_customer_nodes=0,
            num_charge_nodes=1,
            package_weight=0,
            max_energy=-1,
            max_charge_times=0
        )
        obs, info = env.reset()
        # Only depot and one charging station, index 1 is the charging station
        obs, reward, terminated, truncated, info = env.step(1)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

if __name__ == "__main__":
    unittest.main()
