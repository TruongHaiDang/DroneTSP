import numpy as np


def visualize_time_windows(observation):
    nodes = observation["nodes"]
    print(f"\n{'Node':<6} | {'Start Time':<12} | {'End Time':<12} | {'Visited Time':<13} | {'Status'}")
    print("-" * 70)

    for idx, node in enumerate(nodes):
        start_time = float(node[5])
        end_time = float(node[6])
        visited_order = int(node[4])
        visited_time = float(node[7])  # Giả sử node[7] chứa visited_time, cần cập nhật encode

        if not (np.isfinite(start_time) and np.isfinite(end_time)):
            continue

        if visited_order == 0:
            status = "\033[90m─────\033[0m"
            print(f"{idx:<6} | {start_time:<12.5f} | {end_time:<12.5f} | {'-':<13} | {status}")
        else:
            if visited_time < start_time:
                status = "\033[93mĐến sớm\033[0m"
            elif visited_time > end_time:
                status = "\033[91mĐến trễ\033[0m"
            else:
                status = "\033[92mĐúng thời điểm\033[0m"

            print(f"{idx:<6} | {start_time:<12.5f} | {end_time:<12.5f} | {visited_time:<13.5f} | {status}")

def visualize_nodes_status(observation):
    nodes = observation["nodes"]
    print(f"\n{'Node':<6} | {'Lon':<10} | {'Lat':<10} | {'Type':<12} | {'Visited Order':<13} | {'Visited Time':<13} | {'Status'}")
    print("-" * 95)

    for idx, node in enumerate(nodes):
        lon = float(node[0])
        lat = float(node[1])
        node_type = int(node[2])
        visited_order = int(node[4])
        visited_time = float(node[7])

        node_type_str = {0: 'Depot', 1: 'Customer', 2: 'Charging'}.get(node_type, 'Unknown')

        if visited_order == 0:
            status = "\033[97mChưa đi qua\033[0m"  # Trắng
            visited_time_str = "-"
        elif visited_order == 1:
            status = "\033[94mVị trí hiện tại\033[0m"  # Xanh dương
            visited_time_str = f"{visited_time:.5f}"
        else:
            status = f"\033[92mĐã đi (thứ tự {visited_order})\033[0m"  # Xanh lá
            visited_time_str = f"{visited_time:.5f}"

        print(f"{idx:<6} | {lon:<10.5f} | {lat:<10.5f} | {node_type_str:<12} | {visited_order:<13} | {visited_time_str:<13} | {status}")

def visualize_energy_status(observation):
    energy = float(observation["energy_consumption"][0])
    charge_count = int(observation["charge_count"][0])
    total_distance = float(observation["total_distance"][0])
    current_time = float(observation["current_time"][0])

    print("\n\033[95mNĂNG LƯỢNG HIỆN TẠI\033[0m")
    print("\033[95m" + "-" * 25 + "\033[0m")
    print(f"\033[96mNăng lượng tiêu thụ : {energy:.2f}\033[0m")
    print(f"\033[96mSố lần sạc         : {charge_count}\033[0m")
    print(f"\033[96mTổng khoảng cách   : {total_distance:.2f} m\033[0m")
    print(f"\033[96mThời gian hiện tại : {current_time:.2f} s\033[0m")
