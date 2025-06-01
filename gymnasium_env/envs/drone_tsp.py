import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium_env.envs.node_transformer import NodeTransformer
from gymnasium_env.envs.interfaces import NODE_TYPES, Node
from gymnasium_env.envs.utils import calc_energy_consumption, generate_packages_weight
from gymnasium_env.envs.utils import total_distance_of_a_random_route
from geopy.distance import geodesic
from gymnasium_env.envs.folium_exporter import export_to_folium


class DroneTspEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, num_customer_nodes: int = 5, num_charge_nodes: int=1, package_weight: float=40, max_energy: float=-1.0):
        self.num_customer_nodes = num_customer_nodes
        self.num_charge_nodes = num_charge_nodes
        self.max_energy = max_energy # Nếu energy_limit = -1 nghĩa là không quan tâm đến năng lượng.

        # Số 1 là node depot
        total_num_nodes = 1 + self.num_customer_nodes + self.num_charge_nodes
        self.observation_space = spaces.Dict({
            "nodes": spaces.Box(
                low=np.array([-180, -90, 0, 0, 0, 0, 0, 0] * total_num_nodes, dtype=np.float32).reshape(total_num_nodes, -1),
                high=np.array([180, 90, 2, 100, total_num_nodes, float('inf'), float('inf'), float('inf')] * total_num_nodes, dtype=np.float32).reshape(total_num_nodes, -1),
                shape=(total_num_nodes, NodeTransformer.get_shape()),
                dtype=np.float32
            ),
            "total_distance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "energy_consumption": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "current_time": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "charge_count": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int16)
        })

        # Action là index trong danh sách tất cả node.
        self.action_space = spaces.Discrete(n=total_num_nodes, start=0)
        
        # Tổng khoảng cách đã đi
        self.total_distance = 0
        # Năng lượng tiêu thụ
        self.total_energy_consumption = 0
        # Lưu index của node trước đó
        self.prev_position = 0
        # Khối lượng hàng drone đang mang
        self.remain_packages_weight = package_weight
        # Đếm số lần sạc
        self.charge_count = 0
        # Đếm thời gian
        self.current_time = 0.0
        # Cộng dồn số thời gian trễ
        self.late_arrivale_time = 0.0
        # Tốc độ bay của drone, lấy theo DJI Fly-Cart 30
        self.drone_speed = 15 * 3.6 # km/h
        # Lưu trữ giá trị distance và năng lượng giữa các cạnh để tạo input graph
        self.distance_histories = []
        self.energy_consumption_histories = []

        self.__init_nodes()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def __init_nodes(self):
        # Giới hạn vĩ độ và kinh độ cho khu vực TP.HCM
        LAT_BOTTOM, LAT_TOP = 10.75, 10.80
        LON_LEFT, LON_RIGHT = 106.65, 106.72

        # Sinh trọng lượng các gói hàng cho node khách hàng
        packages_weight = generate_packages_weight(
            max_weight=self.remain_packages_weight, total_packages=self.num_customer_nodes
        )

        # === Tạo node Depot ===
        depot_lat = float(self.np_random.uniform(LAT_BOTTOM, LAT_TOP))
        depot_lon = float(self.np_random.uniform(LON_LEFT, LON_RIGHT))
        self.depot = [
            Node(
                lon=depot_lon,  # longitude
                lat=depot_lat,  # latitude
                node_type=NODE_TYPES.depot,
                package_weight=0.0,
                visited_order=1,
                start_time=0.0,
                end_time=float('inf'),
                visited_time=0.0
            )
        ]

        # === Tạo node Khách hàng ===
        self.customer_nodes = []
        for i in range(self.num_customer_nodes):
            # Random vị trí node khách hàng
            lat = float(self.np_random.uniform(LAT_BOTTOM, LAT_TOP))
            lon = float(self.np_random.uniform(LON_LEFT, LON_RIGHT))

            # Thêm node khách hàng vào danh sách
            self.customer_nodes.append(
                Node(
                    lon=lon,
                    lat=lat,
                    node_type=NODE_TYPES.customer,
                    package_weight=float(packages_weight[i]),
                    visited_order=0,
                    start_time=0.0,
                    end_time=0.0,
                    visited_time=0.0
                )
            )

        # === Tạo node Trạm sạc ===
        self.charge_nodes = []
        for _ in range(self.num_charge_nodes):
            lat = float(self.np_random.uniform(LAT_BOTTOM, LAT_TOP))
            lon = float(self.np_random.uniform(LON_LEFT, LON_RIGHT))
            self.charge_nodes.append(
                Node(
                    lon=lon,
                    lat=lat,
                    node_type=NODE_TYPES.charging_station,
                    package_weight=0.0,
                    visited_order=0,
                    start_time=0.0,
                    end_time=float('inf'),
                    visited_time=0.0
                )
            )

        # === Thiết lập khung thời gian cho các node khách hàng một cách tối ưu hơn ===
        random_total_distance = total_distance_of_a_random_route(self.customer_nodes)
        total_time = random_total_distance / self.drone_speed

        RAND_POS_MARGIN = 0.3
        MIN_END_TIME_RATIO = 0.1
        MAX_END_TIME_RATIO = 0.5

        for node in self.customer_nodes:
            # Chọn vị trí bắt đầu ngẫu nhiên trong khoảng [0, 1 - RAND_POS_MARGIN]
            rand_start_ratio = self.np_random.uniform(0, 1 - RAND_POS_MARGIN)
            # Tính thời gian bắt đầu trong khoảng [rand_start_ratio * total_time, (rand_start_ratio + RAND_POS_MARGIN) * total_time]
            start_time = float(self.np_random.uniform(
                rand_start_ratio * total_time,
                (rand_start_ratio + RAND_POS_MARGIN) * total_time
            ))
            # end_time là start_time cộng thêm một khoảng ngẫu nhiên (10% - 50% tổng thời gian)
            end_time = start_time + float(self.np_random.uniform(
                MIN_END_TIME_RATIO * total_time,
                MAX_END_TIME_RATIO * total_time
            ))
            node.start_time = round(start_time, 2)
            node.end_time = round(end_time, 2)

        # Gộp tất cả các node vào danh sách all_nodes
        self.all_nodes = self.depot + self.customer_nodes + self.charge_nodes

    def _get_obs(self):
        nodes_array = np.array([NodeTransformer.encode(node) for node in self.all_nodes], dtype=np.float32)
        return {
            "nodes": nodes_array,
            "total_distance": np.array([self.total_distance], dtype=np.float32),
            "energy_consumption": np.array([self.total_energy_consumption], dtype=np.float32),
            "current_time": np.array([self.current_time], dtype=np.float32),
            "charge_count": np.array([self.charge_count], dtype=np.int16)
        }

    def _get_info(self):
        return {
            "drone_speed": self.drone_speed,
            "customers": self.customer_nodes,
            "distance_histories": self.distance_histories,
            "energy_consumption_histories": self.energy_consumption_histories
        }

    def _sample(self) -> int:
        """
        Trả về index ngẫu nhiên của một node chưa được ghé thăm.
        Dùng để thay thế cho action_space.sample().
        """
        unvisited_indices = [
            idx for idx, node in enumerate(self.all_nodes) if node.visited_order == 0 and node.node_type != NODE_TYPES.charging_station
        ]
        if not unvisited_indices:
            return 0 # Không còn node nào để đi thì trả về vị trí đầu tiên là depot
        return np.random.choice(unvisited_indices)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.__init_nodes()
        self.total_distance = 0
        self.total_energy_consumption = 0
        self.prev_position = 0
        self.remain_packages_weight = 40
        self.current_time = 0.0
        self.charge_count = 0
        self.distance_histories = []
        self.energy_consumption_histories = []

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Action là index của node trong danh sách tất cả node bao gồm khách hàng và trạm sạc.
        prev_node = self.all_nodes[self.prev_position]
        selected_node = self.all_nodes[action]
        # Chỉ cập nhật khi action lớn hơn 0, action bằng 0 là node cuối cùng quay về vị trí 
        # xuất phát, không phải đi đến node mới. Không giới hạn số lần đến trạm sạc.
        if action > 0 and selected_node.node_type != NODE_TYPES.charging_station:
            order = len([node for node in self.all_nodes if node.visited_order > 0])
            selected_node.visited_order = order + 1 # Những node đã đi qua cộng với vị trí đang xét.

        distance = geodesic((prev_node.lat, prev_node.lon), (selected_node.lat, selected_node.lon)).meters
        self.distance_histories.append(distance)
        self.remain_packages_weight -= selected_node.package_weight
        self.total_distance += distance
        energy_consumption = calc_energy_consumption(gij=self.remain_packages_weight)
        self.energy_consumption_histories.append(energy_consumption)
        self.total_energy_consumption += energy_consumption

        # Nếu node này là trạm sạc thì reset mức năng lượng đã tiêu thụ
        if selected_node.node_type == NODE_TYPES.charging_station:
            self.charge_count += 1 # Lưu lại số lần sạc để biết agent có lạm dụng việc sạc hay không.
            self.total_energy_consumption = 0

        # Cập nhật thời gian
        self.current_time += round(distance / self.drone_speed, 2)
        # Gán visited_time để truy hồi trạng thái thời gian của node đã đi qua.
        selected_node.visited_time = self.current_time

        terminated, truncated = False, False
        # Hết năng lượng được xem là truncated. Khi năng lượng tiêu thụ vượt quá mức năng lượng tối đa
        # thì được xem là hết năng lượng.
        if self.max_energy != -1 and self.total_energy_consumption >= self.max_energy:
            truncated = True
        # Luôn bắt đầu từ 0, TSP phải quay về điểm bắt đầu thì mới được xem là hoàn thành.
        if action == 0:
            terminated = True

        if selected_node.node_type == NODE_TYPES.customer:
            if self.current_time < selected_node.start_time:
                self.current_time = selected_node.start_time
            if self.current_time > selected_node.end_time:
                self.late_arrivale_time += self.current_time - selected_node.end_time

        reward = None
        # Chỉ cung cấp reward khi hoàn thành.
        if terminated or truncated:
            # Cung cấp thông tin môi trường để người dùng tự thiết kế reward.
            reward = (self.total_distance, self.total_energy_consumption, self.charge_count, self.late_arrivale_time)

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
        # Tạo danh sách các node đã được ghé thăm theo thứ tự
        visited_nodes = sorted(
            [(idx, n) for idx, n in enumerate(self.all_nodes) if n.visited_order > 0],
            key=lambda x: x[1].visited_order
        )
        path_indices = [idx for idx, _ in visited_nodes]
        if self.prev_position == 0: # Khi hàm step chạy xong thì prev_position cũng chính là action.
            path_indices.append(0) # Nếu như action bằng 0 thì thêm 0 vào cuối để quay về.

        # Xuất bản đồ dạng HTML
        export_to_folium(nodes=self.all_nodes, path_indices=path_indices, file_path="render/index.html")

    def close(self):
        pass
