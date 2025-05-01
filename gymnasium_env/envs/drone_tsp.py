import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from node_encoder import NodeEncoder
from interfaces import NODE_TYPES
from interfaces import Node


class DroneTspEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_customer_nodes: int = 5, num_charge_nodes: int=1):
        self.num_customer_nodes = num_customer_nodes
        self.num_charge_nodes = num_charge_nodes

        total_num_nodes = self.num_customer_nodes + self.num_charge_nodes
        self.observation_space = spaces.Dict(
            {
                # Index trong danh sách tất cả node (khách hàng trước, trạm sạc sau).
                "drone": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int8),
                # Gộp khách hàng và trạm sạc.
                "nodes": spaces.Box(low=-np.inf, high=np.inf, shape=(total_num_nodes, NodeEncoder.get_shape()), dtype=np.float32),
                # Tổng khoảng cách đã đi
                "total_distance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                # Năng lượng tiêu thụ, tính theo paper Trajectory Optimization for Drone Logistics Delivery via Attention-Based Pointer Network.
                "energy_consumption": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            }
        )

        # Action là index trong danh sách tất cả node.
        self.action_space = spaces.Discrete(n=total_num_nodes, start=0)

        self.__init_nodes()

        # Lưu lại vị trí (index trong self.nodes) hiện tại của drone, depot luôn là index 0
        self.drone_position = 0
        # Tổng khoảng cách đã đi
        self.total_distance = 0
        # Năng lượng tiêu thụ
        self.energy_consumption = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def __init_nodes(self):
        COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT = 10, 101

        self.depot = [
            Node(
                x=self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT),
                y=self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT),
                node_type=NODE_TYPES.depot,
                visited_order=0
            )
        ]
        self.customer_nodes = [
            Node(
                x=self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT),
                y=self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT),
                node_type=NODE_TYPES.customer,
                visited_order=0
            ) for _ in range(self.num_customer_nodes)
        ]
        self.charge_nodes = [
            Node(
                x=self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT),
                y=self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT),
                node_type=NODE_TYPES.charging_station,
                visited_order=0
            ) for _ in range(self.num_charge_nodes)
        ]
        self.all_nodes = self.depot + self.customer_nodes + self.charge_nodes

    def _get_obs(self):
        nodes_array = np.array([NodeEncoder.encode(node) for node in self.all_nodes], dtype=np.float32)
        return {
            "drone": np.array([self.drone_position], dtype=np.int8),
            "nodes": nodes_array,
            "total_distance": np.array([self.total_distance], dtype=np.float32),
            "energy_consumption": np.array([self.energy_consumption], dtype=np.float32)
        }

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.__init_nodes()
        self.drone_position = 0
        self.total_distance = 0
        self.energy_consumption = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
