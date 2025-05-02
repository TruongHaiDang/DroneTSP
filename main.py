import gymnasium
import gymnasium_env
from time import sleep

if __name__ == "__main__":
    env = gymnasium.make(
        id="gymnasium_env/DroneTsp-v0",
        render_mode='human',
        num_customer_nodes=5,
        num_charge_nodes=1,
        energy_limit=-1
    )
    observation, info = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()
        print('Action', action)
        observation, reward, terminated, truncated, info = env.step(action=action)
        done = terminated or truncated
        print('Observation', observation)
        print('Reward', reward)
        print('Terminated', terminated)
        print('Truncated', truncated)
        print('Info', info)
        sleep(1)
