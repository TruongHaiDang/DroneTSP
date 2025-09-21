import gymnasium
import gymnasium_env  # Cái này không dùng nhưng bỏ đi thì bị lỗi. Magical.
from time import sleep
from visualize_tools import visualize_energy_status
import re


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def visualize_nodes_status_with_metrics(observation, info):
    """
    In bảng trạng thái các node kèm 2 cột bổ sung:
    - Energy: mức năng lượng tiêu thụ hiện tại ở step này
    - Distance: tổng quãng đường đã đi ở step này

    Các giá trị Energy và Distance là thuộc tính của step (không phải của từng node),
    nên sẽ được hiển thị ở dòng đầu tiên, các dòng còn lại để trống cho hai cột này
    để tránh lặp lại không cần thiết.
    """
    nodes = observation["nodes"]
    # Lịch sử theo từng step
    distance_histories = list(info.get("distance_histories", []))
    energy_histories = list(info.get("energy_consumption_histories", []))

    # Tính tích lũy cho mỗi step
    cum_dist = []
    running = 0.0
    for d in distance_histories:
        running += float(d)
        cum_dist.append(running)

    cum_energy = []
    running_e = 0.0
    for e in energy_histories:
        running_e += float(e)
        cum_energy.append(running_e)

    # Column widths
    w_node = 6
    w_lon = 10
    w_lat = 10
    w_type = 12
    w_weight = 12
    w_visit = 13
    w_status = 24  # Width for visible status text
    w_energy = 10
    w_distance = 12

    # Header
    header = (
        f"\n{'Node':<{w_node}} | {'Lon':<{w_lon}} | {'Lat':<{w_lat}} | "
        f"{'Type':<{w_type}} | {'Weight (kg)':<{w_weight}} | {'Visited Order':<{w_visit}} | "
        f"{'Status':<{w_status}} | {'Energy':<{w_energy}} | {'Distance':<{w_distance}}"
    )
    print(header)
    print("-" * len(_strip_ansi(header)))

    for idx, node in enumerate(nodes):
        lon = float(node[0])
        lat = float(node[1])
        node_type = int(node[2])
        package_weight = float(node[3])
        visited_order = int(node[4])

        node_type_str = {0: "Depot", 1: "Customer", 2: "Charging"}.get(
            node_type, "Unknown"
        )

        if visited_order == 0:
            status = "\033[97mChưa đi qua\033[0m"  # Trắng
        elif visited_order == 1:
            status = "\033[94mVị trí xuất phát\033[0m"  # Xanh dương
        else:
            status = f"\033[92mĐã đi (thứ tự {visited_order})\033[0m"  # Xanh lá

        # Energy/Distance tại step node được ghé (theo visited_order)
        # depot (visited_order=1) => step_index = -1: hiển thị 0
        if visited_order <= 1:
            energy_str = "0.00" if visited_order == 1 else ""
            distance_str = "0.00 m" if visited_order == 1 else ""
        else:
            step_index = visited_order - 2  # mapping: order 2 -> index 0
            if 0 <= step_index < len(cum_energy):
                energy_str = f"{cum_energy[step_index]:.2f}"
            else:
                energy_str = ""
            if 0 <= step_index < len(cum_dist):
                distance_str = f"{cum_dist[step_index]:.2f} m"
            else:
                distance_str = ""

        # Pad colored status based on its visible length to keep columns aligned
        visible_len = len(_strip_ansi(status))
        if visible_len < w_status:
            pad = " " * (w_status - visible_len)
        else:
            pad = ""
        status_cell = f"{status}{pad}"

        line = (
            f"{idx:<{w_node}} | {lon:<{w_lon}.5f} | {lat:<{w_lat}.5f} | {node_type_str:<{w_type}} | "
            f"{package_weight:<{w_weight}.2f} | {visited_order:<{w_visit}} | {status_cell} | {energy_str:<{w_energy}} | {distance_str:<{w_distance}}"
        )
        print(line)


if __name__ == "__main__":
    env = gymnasium.make(
        id="gymnasium_env/DroneTsp-v1",
        render_mode='human',
        num_customer_nodes=5,
        num_charge_nodes=0,
        package_weights=40,
        min_package_weight=5,
        max_package_weight=10,
        max_energy=1000
    )
    observation, info = env.reset()

    done = False
    while not done:
        action = env.unwrapped._sample()
        observation, reward, terminated, truncated, info = env.step(action=action)
        done = terminated or truncated
        print('='*42, 'Action:', action, '='*42)
        visualize_nodes_status_with_metrics(observation, info)
        visualize_energy_status(observation, info)
        sleep(0.5)
