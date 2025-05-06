import gymnasium
import gymnasium_env # Cái này không dùng nhưng bỏ đi thì bị lỗi. Magical.
from time import sleep
from pprint import pprint


if __name__ == "__main__":
    env = gymnasium.make(
        id="gymnasium_env/DroneTsp-v0",
        render_mode='human',
        num_customer_nodes=5,
        num_charge_nodes=1,
        max_energy=-1
    )
    observation, info = env.reset()

    done = False
    while not done:
        action = env.unwrapped._sample()
        print('-'*100)
        print('Action:', action)
        observation, reward, terminated, truncated, info = env.step(action=action)
        done = terminated or truncated
        print('Observation:')
        pprint(observation)
        print('Reward:', reward)
        print('Terminated:', terminated)
        print('Truncated:', truncated)
        print('Info')
        pprint(info)
        sleep(0.5)
