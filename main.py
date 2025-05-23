import gymnasium
import gymnasium_env # Cái này không dùng nhưng bỏ đi thì bị lỗi. Magical.
from time import sleep
from pprint import pprint
from visualize_tools import visualize_time_windows, visualize_nodes_status, visualize_energy_status


if __name__ == "__main__":
    env = gymnasium.make(
        id="gymnasium_env/DroneTsp-v1",
        render_mode='human',
        num_customer_nodes=5,
        num_charge_nodes=1,
        package_weight=40,
        max_energy=-1
    )
    observation, info = env.reset()

    done = False
    while not done:
        action = env.unwrapped._sample()
        observation, reward, terminated, truncated, info = env.step(action=action)
        done = terminated or truncated
        print('='*50, 'Action:', action, '='*50)
        visualize_nodes_status(observation)
        visualize_time_windows(observation)
        visualize_energy_status(observation)
        sleep(0.5)
