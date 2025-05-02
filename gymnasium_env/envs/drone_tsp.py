import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium_env.envs.node_encoder import NodeEncoder
from gymnasium_env.envs.interfaces import NODE_TYPES, Node
from gymnasium_env.envs.utils import euclidean_distance, calc_energy_consumption, generate_packages_weight
import random


class DroneTspEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_customer_nodes: int = 5, num_charge_nodes: int=1, energy_limit: float=-1.0):
        self.num_customer_nodes = num_customer_nodes
        self.num_charge_nodes = num_charge_nodes
        self.energy_limit = energy_limit # Nếu energy_limit = -1 nghĩa là không quan tâm đến năng lượng.

        # Số 1 là node depot
        total_num_nodes = 1 + self.num_customer_nodes + self.num_charge_nodes
        self.observation_space = spaces.Dict(
            {
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
        
        # Tổng khoảng cách đã đi
        self.total_distance = 0
        # Năng lượng tiêu thụ
        self.energy_consumption = 0
        # Lưu index của node trước đó
        self.prev_position = 0
        # Khối lượng hàng drone đang mang
        self.remain_packages_weight = 40

        self.__init_nodes()

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
        if self.render_mode == "human":
            self.screen_width = 1920
            self.screen_height = 1080

    def __init_nodes(self):
        COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT = 10, 101
        packages_weight = generate_packages_weight(max_weight=self.remain_packages_weight, total_packages=self.num_customer_nodes)

        self.depot = [
            Node(
                x=int(self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT)),
                y=int(self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT)),
                node_type=NODE_TYPES.depot,
                package_weight=0.0,
                visited_order=1
            )
        ]
        self.customer_nodes = [
            Node(
                x=int(self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT)),
                y=int(self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT)),
                node_type=NODE_TYPES.customer,
                package_weight=float(packages_weight[i]),
                visited_order=0
            ) for i in range(self.num_customer_nodes)
        ]
        self.charge_nodes = [
            Node(
                x=int(self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT)),
                y=int(self.np_random.integers(COOR_BOTTOM_LIMIT, COOR_TOP_LIMIT)),
                node_type=NODE_TYPES.charging_station,
                package_weight=0.0,
                visited_order=0
            ) for _ in range(self.num_charge_nodes)
        ]
        self.all_nodes = self.depot + self.customer_nodes + self.charge_nodes

    def _get_obs(self):
        nodes_array = np.array([NodeEncoder.encode(node) for node in self.all_nodes], dtype=np.float32)
        return {
            "nodes": nodes_array,
            "total_distance": np.array([self.total_distance], dtype=np.float32),
            "energy_consumption": np.array([self.energy_consumption], dtype=np.float32)
        }

    def _get_info(self):
        return {}

    def sample(self) -> int:
        """
        Trả về index ngẫu nhiên của một node chưa được ghé thăm.
        Dùng để thay thế cho action_space.sample().
        """
        unvisited_indices = [
            idx for idx, node in enumerate(self.all_nodes) if node.visited_order == 0
        ]
        if not unvisited_indices:
            return 0 # Không còn node nào để đi thì trả về vị trí đầu tiên là depot
        return self.np_random.choice(unvisited_indices)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.__init_nodes()
        self.total_distance = 0
        self.total_energy_consumption = 0
        self.prev_position = 0
        self.remain_packages_weight = 40

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Action là index của node trong danh sách tất cả node bao gồm khách hàng và trạm sạc.
        prev_node = self.all_nodes[self.prev_position]
        selected_node = self.all_nodes[action]

        distance = euclidean_distance(node_1=prev_node, node_2=selected_node)
        self.remain_packages_weight -= selected_node.package_weight
        self.total_distance += distance
        energy_consumption = calc_energy_consumption(gij=self.remain_packages_weight)
        self.total_energy_consumption += energy_consumption
        
        terminated, truncated = False, False
        # Hết năng lượng được xem là truncated.
        if self.energy_limit != -1 and self.total_energy_consumption >= self.energy_limit:
            truncated = True
        # Luôn bắt đầu từ 0, TSP phải quay về điểm bắt đầu thì mới được xem là hoàn thành.
        if action == 0:
            terminated = True
        
        reward = 0
        # Chỉ cung cấp reward khi hoàn thành lộ trình.
        if terminated or truncated:
            # Đi càng xa càng bị phạt, dùng càng nhiều năng lượng càng bị phạt.
            reward = -self.total_distance -self.total_energy_consumption

        observation = self._get_obs()
        info = self._get_info()

        # Đánh dấu là node trước đó sau khi hoàn thành xử lý
        self.prev_position = action

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((255, 255, 255))
        
        

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
