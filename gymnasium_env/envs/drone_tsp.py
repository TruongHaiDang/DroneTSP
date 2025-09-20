import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium_env.envs.node_transformer import NodeTransformer
from gymnasium_env.envs.interfaces import NODE_TYPES, Node
from gymnasium_env.envs.utils import calc_energy_consumption
from gymnasium_env.envs.utils import total_distance_of_a_random_route
from geopy.distance import geodesic
from gymnasium_env.envs.folium_exporter import export_to_folium


class DroneTspEnv(gym.Env):
    """Mô phỏng môi trường drone giao hàng dựa trên TSP.

    Args:
        gym (gym.Env): Kế thừa lớp Env của gymnasium

    Returns:
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        num_customer_nodes: int = 5,
        num_charge_nodes: int = 1,
        package_weights: float = 40,
        min_package_weight: float = 1,
        max_package_weight: float = 5,
        max_energy: float = -1.0,
        max_charge_times: int = -1,
    ):
        """Constructor của class

        Args:
            render_mode (str, optional): Loại hiển thị. Defaults to None.
            num_customer_nodes (int, optional): Số lượng node nhận hàng. Defaults to 5.
            num_charge_nodes (int, optional): Số lượng trạm sạc. Defaults to 1.
            package_weights (float, optional): Sức chứa tối đa drone có thể mang (kg). Defaults to 40.
            min_package_weight (float, optional): Khối lượng tối thiểu mỗi đơn hàng (kg). Defaults to 1.
            max_package_weight (float, optional): Khối lượng tối đa mỗi đơn hàng (kg). Defaults to 5.
            max_energy (float, optional): Tổng năng lượng của drone. Defaults to -1.0.
            max_charge_times (int, optional): Số lần sạc tối đa của drone. Giá trị âm để bỏ giới hạn.
        """
        self.num_customer_nodes = num_customer_nodes
        self.num_charge_nodes = num_charge_nodes
        self.min_package_weight_per_node = min_package_weight
        self.max_package_weight_per_node = max_package_weight
        self.max_energy = (
            max_energy  # Nếu energy_limit = -1 nghĩa là không quan tâm đến năng lượng.
        )
        self.max_charge_times = max_charge_times
        # Số 1 là node depot
        total_num_nodes = 1 + self.num_customer_nodes + self.num_charge_nodes
        self.observation_space = spaces.Dict(
            {
                "nodes": spaces.Box(
                    low=np.array(
                        [-180, -90, 0, 0, 0] * total_num_nodes, dtype=np.float32
                    ).reshape(total_num_nodes, -1),
                    high=np.array(
                        [180, 90, 2, 100, total_num_nodes] * total_num_nodes,
                        dtype=np.float32,
                    ).reshape(total_num_nodes, -1),
                    shape=(total_num_nodes, NodeTransformer.get_shape()),
                    dtype=np.float32,
                ),
                "total_distance": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "energy_consumption": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "charge_count": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.int16
                ),
            }
        )

        # Action là index trong danh sách tất cả node.
        self.action_space = spaces.Discrete(n=total_num_nodes, start=0)
        # Tổng khoảng cách đã đi
        self.total_distance = 0
        # Năng lượng tiêu thụ
        self.total_energy_consumption = 0
        # Lưu index của node trước đó
        self.prev_position = 0
        # Khối lượng hàng drone có thể mang tối đa (sức chứa)
        self.max_packages_weight = package_weights
        self.remain_packages_weight = self.max_packages_weight
        # Đếm số lần sạc
        self.charge_count = 0
        # Tốc độ bay của drone, lấy theo DJI Fly-Cart 30
        self.drone_speed = 15  # m/s
        # Lưu trữ giá trị distance và năng lượng giữa các cạnh để tạo input graph
        self.distance_histories = []
        self.energy_consumption_histories = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def __init_nodes(self):
        """Khởi tạo danh sách node"""
        # Giới hạn vĩ độ và kinh độ cho khu vực TP.HCM
        LAT_BOTTOM, LAT_TOP = 10.75, 10.80
        LON_LEFT, LON_RIGHT = 106.65, 106.72

        # Sinh trọng lượng từng gói hàng cho node khách hàng (độc lập)
        packages_weight = [
            float(self.np_random.uniform(self.min_package_weight_per_node, self.max_package_weight_per_node))
            for _ in range(self.num_customer_nodes)
        ]

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
                )
            )

        # === Tạo node Trạm sạc ===
        self.charge_nodes = []
        for i in range(self.num_charge_nodes):
            lat = float(self.np_random.uniform(LAT_BOTTOM, LAT_TOP))
            lon = float(self.np_random.uniform(LON_LEFT, LON_RIGHT))
            self.charge_nodes.append(
                Node(
                    lon=lon,
                    lat=lat,
                    node_type=NODE_TYPES.charging_station,
                    package_weight=0.0,
                    visited_order=0,
                )
            )

        # Gộp tất cả các node vào danh sách all_nodes
        self.all_nodes = self.depot + self.customer_nodes + self.charge_nodes

    def _get_obs(self):
        """Định nghĩa observation của môi trường

        Returns:
            obs: Observation
        """
        nodes_array = np.array(
            [NodeTransformer.encode(node) for node in self.all_nodes], dtype=np.float32
        )
        return {
            "nodes": nodes_array,
            "total_distance": np.array([self.total_distance], dtype=np.float32),
            "energy_consumption": np.array(
                [self.total_energy_consumption], dtype=np.float32
            ),
            "charge_count": np.array([self.charge_count], dtype=np.int16),
        }

    def _get_info(self):
        """Cung cấp thông tin bổ sung của môi trường

        Returns:
            infor: Thông tin bổ sung của môi trường
        """
        return {
            "drone_speed": self.drone_speed,
            "customers": self.customer_nodes,
            "distance_histories": self.distance_histories,
            "energy_consumption_histories": self.energy_consumption_histories,
            "charge_count": self.charge_count,
            "remain_packages_weight": self.remain_packages_weight,
            "max_energy": self.max_energy,
        }

    def _sample(self) -> int:
        """
        Trả về index ngẫu nhiên của một node chưa được ghé thăm.
        Dùng để thay thế cho action_space.sample().
        """
        unvisited_indices = [
            idx
            for idx, node in enumerate(self.all_nodes)
            if node.visited_order == 0 and node.node_type != NODE_TYPES.charging_station
        ]
        if not unvisited_indices:
            return 0  # Không còn node nào để đi thì trả về vị trí đầu tiên là depot
        return np.random.choice(unvisited_indices)

    def reset(self, seed=None, options=None):
        """Reset môi trường

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            obs: Observation của môi trường
            info: Thông tin phụ của môi trường
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if options is None:
            options = {}
        new_coordinates: bool = bool(options.get("new_coordinates", True))

        self.total_distance = 0
        self.total_energy_consumption = 0
        self.prev_position = 0
        self.remain_packages_weight = self.max_packages_weight
        self.charge_count = 0
        self.distance_histories = []
        self.energy_consumption_histories = []
        if new_coordinates == True:
            self.__init_nodes()
        else:
            for node in self.all_nodes:
                if node.node_type == NODE_TYPES.depot:
                    node.visited_order = 1
                elif (
                    node.node_type == NODE_TYPES.customer
                    or node.node_type == NODE_TYPES.charging_station
                ):
                    node.visited_order = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Thực hiện một bước trong môi trường với hành động được cung cấp.

        Args:
            action (int): Chỉ số của node sẽ được ghé thăm tiếp theo trong danh sách all_nodes. Có thể là node khách hàng, trạm sạc hoặc depot (chỉ số 0).

        Returns:
            observation (dict): Quan sát hiện tại của môi trường sau khi thực hiện hành động.
            reward (tuple hoặc None): Bộ giá trị thưởng (tổng quãng đường, tổng năng lượng tiêu thụ, số lần sạc, tổng thời gian trễ) nếu kết thúc episode, ngược lại là None.
            terminated (bool): True nếu episode kết thúc do quay về depot, ngược lại là False.
            truncated (bool): True nếu episode kết thúc do vượt quá giới hạn năng lượng, ngược lại là False.
            info (dict): Thông tin bổ sung về trạng thái môi trường.
        """
        terminated, truncated = False, False
        # Action là index của node trong danh sách tất cả node bao gồm khách hàng và trạm sạc.
        prev_node = self.all_nodes[self.prev_position]
        selected_node = self.all_nodes[action]
        # Chỉ cập nhật khi action lớn hơn 0, action bằng 0 là node cuối cùng quay về vị trí
        # xuất phát, không phải đi đến node mới. Không giới hạn số lần đến trạm sạc.
        distance = geodesic(
            (prev_node.lat, prev_node.lon), (selected_node.lat, selected_node.lon)
        ).meters
        self.distance_histories.append(distance)
        if action > 0 and selected_node.node_type != NODE_TYPES.charging_station:
            self.remain_packages_weight -= selected_node.package_weight
            if self.remain_packages_weight < 0:
                # Không để khối lượng còn lại âm để tránh lỗi tính năng lượng
                truncated = True
            order = len([node for node in self.all_nodes if node.visited_order > 0])
            selected_node.visited_order = (
                order + 1
            )  # Những node đã đi qua cộng với vị trí đang xét.
        self.total_distance += distance
        energy_consumption = calc_energy_consumption(
            gij=self.remain_packages_weight,
            distanceij=distance,
            speedij=self.drone_speed,
        )
        self.energy_consumption_histories.append(energy_consumption)
        self.total_energy_consumption += energy_consumption

        # Nếu node này là trạm sạc thì reset mức năng lượng đã tiêu thụ
        if selected_node.node_type == NODE_TYPES.charging_station:
            self.charge_count += (
                1  # Lưu lại số lần sạc để biết agent có lạm dụng việc sạc hay không.
            )
            self.total_energy_consumption = 0

        # Luôn bắt đầu từ 0, TSP phải quay về điểm bắt đầu thì mới được xem là hoàn thành.
        if action == 0:
            self.charge_count += 1
            self.total_energy_consumption = 0
            # Quay về depot để lấy thêm hàng: nạp lại sức chứa
            self.remain_packages_weight = self.max_packages_weight
            truncated = True

        # Hết năng lượng được xem là truncated. Khi năng lượng tiêu thụ vượt quá mức năng lượng tối đa
        # thì được xem là hết năng lượng.
        if self.max_energy != -1 and self.total_energy_consumption >= self.max_energy:
            truncated = True
        if self.max_charge_times != -1 and self.charge_count > self.max_charge_times:
            truncated = True

        if action == 0 and all(
            node.visited_order > 0
            for node in self.all_nodes
            if node.node_type != NODE_TYPES.charging_station
        ):
            terminated = True

        observation = self._get_obs()
        info = self._get_info()

        # Đánh dấu là node trước đó sau khi hoàn thành xử lý
        self.prev_position = action

        if self.render_mode == "human":
            self._render_frame()

        return observation, distance, terminated, truncated, info

    def render(self):
        """
        Hiển thị môi trường theo chế độ render_mode đã chọn.

        Nếu render_mode là 'rgb_array', trả về frame đã render dưới dạng mảng.
        Nếu render_mode là 'human', hiển thị trực quan môi trường (xử lý trong _render_frame).
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Phương thức nội bộ để hiển thị trạng thái hiện tại của môi trường.

        Sinh bản đồ HTML trực quan hóa đường đi và các node đã ghé thăm bằng folium,
        và lưu vào 'render/index.html'.
        """
        # Tạo danh sách các node đã được ghé thăm theo thứ tự
        visited_nodes = sorted(
            [(idx, n) for idx, n in enumerate(self.all_nodes) if n.visited_order > 0],
            key=lambda x: x[1].visited_order,
        )
        path_indices = [idx for idx, _ in visited_nodes]
        if (
            self.prev_position == 0
        ):  # Khi hàm step chạy xong thì prev_position cũng chính là action.
            path_indices.append(
                0
            )  # Nếu như action bằng 0 thì thêm 0 vào cuối để quay về.

        # Xuất bản đồ dạng HTML
        export_to_folium(
            nodes=self.all_nodes,
            path_indices=path_indices,
            file_path="render/index.html",
        )

    def close(self):
        pass
