import folium
from folium import Map, PolyLine, Marker
from gymnasium_env.envs.interfaces import NODE_TYPES
import os


def export_to_folium(nodes: list, path_indices: list, file_path="render/index.html"):
    """
    Tạo bản đồ Folium từ danh sách node và đường đi đã chọn.
    """
    # Tạo thư mục cha nếu chưa có
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not nodes or not path_indices:
        raise ValueError("Phải có danh sách node và path.")

    # Lấy trung tâm bản đồ là depot
    depot = nodes[0]
    m = folium.Map(location=[depot.lat, depot.lon], zoom_start=14, tiles="OpenStreetMap")

    # Thêm marker cho từng node
    for i, node in enumerate(nodes):
        color = (
            "red" if node.node_type == NODE_TYPES.depot else
            "blue" if node.node_type == NODE_TYPES.charging_station else
            "green"
        )
        label = f"{i} ({node.node_type.name})"
        Marker(
            location=(node.lat, node.lon),
            popup=label,
            icon=folium.Icon(color=color)
        ).add_to(m)

    # Thêm polyline cho route đã đi
    latlon_path = [(nodes[i].lat, nodes[i].lon) for i in path_indices]
    PolyLine(locations=latlon_path, color="blue", weight=5).add_to(m)

    m.save(file_path)
