# DroneTSP

Môi trường drone giao hàng dựa trên bài toán TSP. Môi trường này dùng cho các dự án học tăng cường.

## Môi trường

Kho lưu trữ này lưu trữ các ví dụ được hiển thị [trong tài liệu tạo môi trường](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

- `DroneTspEnv`: Môi trường drone giao hàng dựa trên bài toán TSP.

## Bộ bao

Kho lưu trữ này lưu trữ các ví dụ được hiển thị [trong tài liệu bộ bao](https://gymnasium.farama.org/api/wrappers/).

- `ClipReward`: Một `RewardWrapper` cắt giảm phần thưởng ngay lập tức vào một phạm vi hợp lệ
- `DiscreteActions`: Một `ActionWrapper` giới hạn không gian hành động vào một tập hợp con hữu hạn
- `RelativePosition`: Một `ObservationWrapper` tính toán vị trí tương đối giữa một tác nhân và một mục tiêu
- `ReacherRewardWrapper`: Cho phép chúng ta cân nhắc các điều khoản phần thưởng cho môi trường reacher

## Đóng góp

Nếu bạn muốn đóng góp, hãy làm theo các bước sau:

- Fork kho lưu trữ này
- Clone fork của bạn
- Cài đặt pre-commit qua `pre-commit install`

PRs có thể yêu cầu PRs đi kèm trong [kho tài liệu](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).

## Cài đặt

Để cài đặt môi trường mới của bạn, hãy chạy các lệnh sau:

```bash
cd gymnasium_env
pip install -e .
```

# Đặc tả hệ thống và hướng dẫn sử dụng

## 🚁 DroneTSP

- **Mục tiêu**:
  Lập lộ trình tối ưu cho drone giao hàng từ kho đến nhiều khách hàng, sử dụng ít năng lượng nhất, không hết pin giữa đường và quay về kho an toàn. Có thể sử dụng trạm sạc nhưng bị phạt nếu lạm dụng.

- **Không gian hành động**: `Discrete(N)`
  `N = 1 + num_customer_nodes + num_charge_nodes`.
  Mỗi action tương ứng với chỉ số của node trong danh sách các node.

  - `0`: Kho (Depot) – bắt buộc quay về cuối hành trình.
  - `1..num_customer_nodes`: Các khách hàng.
  - `num_customer_nodes+1..N-1`: Các trạm sạc.

- **Không gian quan sát**: `Dict` gồm:

  - `nodes`: `Box(shape=(N, 5))`
    Mỗi node được mã hóa thành `[lon, lat, node_type, package_weight, visited_order]`.
  - `total_distance`: Tổng quãng đường đã đi.
  - `energy_consumption`: Năng lượng đã tiêu thụ.

- **Phần thưởng**:

  - Chỉ được cung cấp khi kết thúc episode (terminated hoặc truncated).
  - Công thức:

    - Nếu thành công (quay về depot):
      `reward = -distance - energy - 10 * số lần sạc`
    - Nếu thất bại (hết năng lượng):
      `reward = -1000 - distance - energy - 10 * số lần sạc`

- **Tiêu chí kết thúc**:

  - `terminated = True` khi agent chọn action = 0 (tức quay về depot).
  - `truncated = True` khi năng lượng tiêu thụ vượt quá `max_energy`.

- **Đặc điểm nổi bật**:

  - Mô phỏng thực tế với bản đồ địa lý khu vực TP.HCM.
  - Các node được tạo ngẫu nhiên trong khoảng tọa độ thực.
  - Trọng lượng hàng được sinh để tổng không vượt quá sức chở drone (40kg).
  - Mức năng lượng giới hạn có thể tùy chỉnh hoặc vô hạn (`max_energy = -1`).
  - Môi trường phù hợp để thử nghiệm thuật toán: Q-learning, GNN, A3C, PPO,...

- **Chế độ hiển thị**:

  - `render_mode='human'`: Xuất bản đồ HTML trực quan với đường đi, node.
  - Bản đồ được lưu tại `render/index.html` sau mỗi bước.

- **Cách sử dụng**:

  ```python
  import gymnasium

  env = gymnasium.make(
      id="gymnasium_env/DroneTsp-v0",
      render_mode="human",
      num_customer_nodes=5,
      num_charge_nodes=1,
      max_energy=50000.0  # hoặc -1 để bỏ giới hạn năng lượng
  )

  observation, info = env.reset()
  done = False
  while not done:
      action = env.unwrapped._sample()  # Lấy ngẫu nhiên node chưa đi
      observation, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
  ```
