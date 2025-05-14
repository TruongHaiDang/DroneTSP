import random
from geopy.distance import geodesic


def generate_packages_weight(max_weight: float, total_packages: int):
    if max_weight == 0 or total_packages == 0:
        return []

    result = []

    # Tạo danh sách điểm cắt
    cut_points = sorted([random.randint(0, max_weight) for _ in range(total_packages - 1)])
    cut_points = [0] + cut_points + [max_weight]

    # Tính độ dài giữa các điểm cắt
    result = [round(cut_points[i+1] - cut_points[i]) for i in range(total_packages)]

    # Điều chỉnh nết tổng vượt quá max_weight
    diff = sum(result) - max_weight
    while diff != 0:
        for i in range(len(result)):
            if diff == 0:
                break
            if diff > 0 and result[i] > 0:
                result[i] -= 1
                diff -= 1
            elif diff < 0:
                result[i] += 1
                diff += 1

    return result

def calc_energy_consumption(gij: float):
    """Tính năng lượng tiêu thụ, hàm này theo công thức trong bài báo 
    Trajectory Optimization for Drone Logistics
    Delivery via Attention-Based Pointer Network

    Args:
        gij (float): Khối lượng hàng drone phải mang giữa hai điểm i và j

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

    return (total_mass ** 1.5) * lambda_coef

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
    return total_distance
