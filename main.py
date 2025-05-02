import gymnasium
import gymnasium_env


if __name__ == "__main__":
    env = gymnasium.make(
        id="gymnasium_env/DroneTsp-v0",
        render_mode='human',
        num_customer_nodes=5,
        num_charge_nodes=1,
        energy_limit=-1
    )
    observation, info = env.reset()

    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     print(action)
    #     done = True
