# DroneTSP

Môi trường Drone TSP cho Gymnasium mô phỏng drone giao hàng trong thành phố (TP.HCM), với các nút: kho (depot), khách hàng và trạm sạc. Toàn bộ đặc tả dưới đây được tổng hợp từ mã nguồn trong thư mục `gymnasium_env/envs`.

## Cài đặt

```bash
pip install -e .
```

Sau cài đặt, môi trường được đăng ký ở id: `gymnasium_env/DroneTsp-v1`.

## Sử dụng nhanh

```python
import gymnasium as gym

env = gym.make(
    id="gymnasium_env/DroneTsp-v1",
    render_mode="human",         # "human" hoặc "rgb_array"
    num_customer_nodes=5,
    num_charge_nodes=1,
    package_weight=40,            # sức chứa tối đa (kg)
    min_package_weight=1,
    max_package_weight=5,
    max_energy=-1.0,              # âm để bỏ giới hạn năng lượng
    max_charge_times=-1           # âm để bỏ giới hạn số lần sạc
)

obs, info = env.reset(options={"new_coordinates": True})
done = False
while not done:
    action = env.unwrapped._sample()  # Lấy ngẫu nhiên node khách hàng chưa đi
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

## DroneTspEnv

- Mã nguồn: `gymnasium_env/envs/drone_tsp.py`
- Đăng ký: `gymnasium_env/DroneTsp-v1`

### Tham số khởi tạo

- `render_mode`: `None | "human" | "rgb_array"`
- `num_customer_nodes` (int): số khách hàng (mặc định 5)
- `num_charge_nodes` (int): số trạm sạc (mặc định 1)
- `package_weight` (float): sức chứa tối đa của drone (kg, mặc định 40)
- `min_package_weight` (float): khối lượng đơn tối thiểu (kg, mặc định 1)
- `max_package_weight` (float): khối lượng đơn tối đa (kg, mặc định 5)
- `max_energy` (float): ngưỡng năng lượng tiêu thụ; `-1` để bỏ giới hạn
- `max_charge_times` (int): số lần nạp năng lượng tối đa; âm để bỏ giới hạn

### Không gian quan sát (`observation_space`)

- `Dict` gồm:
  - `nodes`: `Box(shape=(N, 5), dtype=float32)` với `N = 1 + num_customer_nodes + num_charge_nodes`
    - Mỗi node được mã hóa: `[lon, lat, node_type, package_weight, visited_order]`
    - `node_type`: 0=depot, 1=customer, 2=charging_station
  - `total_distance`: tổng quãng đường đã đi (m)
  - `energy_consumption`: năng lượng tiêu thụ tích lũy hiện tại
  - `charge_count`: số lần sạc đã thực hiện

### Không gian hành động (`action_space`)

- `Discrete(N, start=0)` với cùng `N` như trên
- `0`: depot; `1..num_customer_nodes`: khách hàng; còn lại: trạm sạc

### Quy tắc bước (`step`)

- Trả về: `(observation, reward, terminated, truncated, info)`
- `reward`: là khoảng cách (m) của cạnh vừa di chuyển (theo `geodesic`) cho hành động hiện tại.
- Khi đi đến khách hàng: giảm `remain_packages_weight` theo `package_weight` của node đó; ghi `visited_order`.
- Khi đến trạm sạc: tăng `charge_count` và đặt lại `total_energy_consumption` về 0.
- Khi `action == 0` (đi về depot):
  - tăng `charge_count`, đặt lại `total_energy_consumption` về 0
  - nạp lại sức chứa: `remain_packages_weight = package_weight`
  - `truncated = True` (kết thúc sớm một vòng hành trình)
- Điều kiện dừng:
  - `terminated = True` nếu `action == 0` và tất cả các node không phải trạm sạc đã được ghé (`visited_order > 0`)
  - `truncated = True` nếu vượt `max_energy` hoặc vượt `max_charge_times` hoặc khi `action == 0`

### Reset

- `env.reset(seed=None, options=None)`:
  - `options["new_coordinates"] = True` (mặc định): tạo lại toàn bộ vị trí các node (TP.HCM trong khung [10.75–10.80] x [106.65–106.72])
  - Nếu `False`: giữ nguyên toạ độ cũ, đặt lại `visited_order` và trạng thái tích lũy

### Render

- `render_mode="human"`: sinh bản đồ HTML ở `render/index.html` bằng `folium`, hiển thị đường đi theo thứ tự `visited_order`
- `render_mode="rgb_array"`: trả về frame từ `_render_frame()` (không vẽ GUI ngoài)

### Trường thông tin (`info`)

- `drone_speed` (m/s), `customers` (danh sách node khách hàng),
  `distance_histories`, `energy_consumption_histories`, `charge_count`,
  `remain_packages_weight`, `max_energy`.

## Các mô-đun trong `envs/`

### `interfaces.py`

- `NODE_TYPES`: Enum các loại node: `depot=0`, `customer=1`, `charging_station=2`
- `Node`: dataclass gồm `lon, lat, node_type, package_weight, visited_order`

### `node_transformer.py`

- `NodeTransformer.encode(Node) -> np.ndarray[5]`: mã hoá Node thành mảng 5 phần tử
- `NodeTransformer.decode(arr) -> Node`: giải mã về Node
- `NodeTransformer.get_shape() -> int`: kích thước vector nút (=5)

### `utils.py`

- `generate_packages_weight(max_weight, total_packages)`:
  sinh danh sách khối lượng nguyên, tổng xấp xỉ `max_weight`
- `calc_energy_consumption(gij, distanceij, speedij=15)`:
  tính năng lượng tiêu thụ cho cạnh theo công thức trong bài báo; đầu ra làm tròn 2 chữ số
- `total_distance_of_a_random_route(nodes)`:
  tổng quãng đường qua danh sách node theo thứ tự
- `calc_distance(node_a, node_b)`:
  khoảng cách địa lý giữa hai điểm `[lon, lat]`

### `folium_exporter.py`

- `export_to_folium(nodes, path_indices, file_path="render/index.html")`:
  vẽ map, đánh dấu màu theo loại node và vẽ `Polyline` theo `path_indices`

## Ghi chú

- Id môi trường đúng là `gymnasium_env/DroneTsp-v1` (được đăng ký tại `gymnasium_env/__init__.py`).
- `reward` hiện là khoảng cách bước đi; nếu bạn muốn phần thưởng tập trung mục tiêu (ví dụ nhỏ hoá tổng quãng đường, năng lượng, số lần sạc), bạn có thể điều chỉnh trong `step()` để trả về giá trị phù hợp.

## Đóng góp

- Cài đặt `pre-commit` và chạy `pre-commit install`
- Mở PR với mô tả rõ thay đổi và cách kiểm thử
