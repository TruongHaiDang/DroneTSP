import random
from geopy.distance import geodesic


def generate_packages_weight(max_weight: float, total_packages: int):
    """
    Ngẫu nhiên tạo ra một danh sách khối lượng các gói hàng sao cho tổng khối lượng xấp xỉ max_weight.
    Mỗi khối lượng gói hàng là số nguyên không âm, và số lượng gói là total_packages.

    Tham số:
        max_weight (float): Tổng khối lượng tối đa cần phân phối cho các gói hàng.
        total_packages (int): Số lượng gói hàng cần tạo khối lượng.

    Trả về:
        list[int]: Danh sách các số nguyên đại diện cho khối lượng từng gói hàng, tổng lại bằng max_weight.
    """
    if max_weight < 0 or total_packages < 0:
        raise ValueError("Max weight and total packages can't be negative.")
    
    if max_weight == 0 or total_packages == 0:
        return []

    result = []

    # Tạo danh sách điểm cắt ngẫu nhiên đã sắp xếp để chia khối lượng
    cut_points = sorted([random.randint(0, max_weight) for _ in range(total_packages - 1)])
    cut_points = [0] + cut_points + [max_weight]

    # Tính hiệu giữa các điểm cắt liên tiếp để lấy khối lượng từng gói
    result = [round(cut_points[i+1] - cut_points[i]) for i in range(total_packages)]

    # Điều chỉnh nếu tổng khối lượng chưa đúng max_weight
    diff = sum(result) - max_weight
    while diff != 0:
        for i in range(len(result)):
            if diff == 0:
                break
            if diff > 0 and result[i] > 0:
                result[i] -= 1  # Giảm khối lượng nếu tổng vượt quá max_weight
                diff -= 1
            elif diff < 0:
                result[i] += 1  # Tăng khối lượng nếu tổng nhỏ hơn max_weight
                diff += 1

    return result

def calc_energy_consumption(gij: float, distanceij: float, speedij: float = 15):
    """Tính năng lượng tiêu thụ, hàm này theo công thức trong bài báo 
    Trajectory Optimization for Drone Logistics
    Delivery via Attention-Based Pointer Network

    Args:
        gij (float): Khối lượng hàng drone phải mang giữa hai điểm i và j (kg).
        distanceij (float): Khoảng cách giữa 2 điểm i và j (m).
        speedij (float): Tốc độ bay của drone giữa 2 điểm i và j (m/s).

    Returns:
        float: Năng lượng tiêu thụ
    """
    if gij < 0:
        raise ValueError("Weight can't be negative.")

    drone_frame_weight = 42.5  # kg
    battery_weight = 22.5      # kg
    gravity = 9.81             # m/s^2
    wind_fluid_density = 1.225 # kg/m^3
    motor_area = 1.375         # m^2
    motor_number = 8

    total_mass = drone_frame_weight + battery_weight + gij
    lambda_coef = (gravity ** 3) / (2 * wind_fluid_density * motor_area * motor_number)

    gij_energy_consumption = (total_mass ** 1.5) * lambda_coef
    gij_energy_consumption /= 1_000.0
    distanceij_energy_consumption = distanceij / 100.0
    energy_consumption = gij_energy_consumption * (distanceij_energy_consumption / speedij)
    return round(energy_consumption, 2)

def total_distance_of_a_random_route(nodes):
    """
    Tính tổng khoảng cách đi qua tất cả các node trong danh sách (thứ tự giữ nguyên).
    Tham số:
        nodes: list các node (phải có thuộc tính lat, lon)
    Kết quả:
        Tổng khoảng cách (mét)
    """
    if len(nodes) < 2:
        return 0.0
    total_distance = 0.0
    for i in range(len(nodes) - 1):
        node_a = nodes[i]
        node_b = nodes[i + 1]
        total_distance += geodesic((node_a.lat, node_a.lon), (node_b.lat, node_b.lon)).meters
    return round(total_distance, 2)

def calc_distance(node_a, node_b):
    distance = geodesic((node_a[1], node_a[0]), (node_b[1], node_b[0])).meters
    return round(distance, 2)
