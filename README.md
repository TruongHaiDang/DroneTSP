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
