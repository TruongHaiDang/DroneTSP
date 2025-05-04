from staticmap import StaticMap, CircleMarker
import pygame
from io import BytesIO


class TspMap:
    def __init__(self, width=1920, height=1080, center=(10.7769, 106.7009), zoom=15):
        """
        center: tuple(lat, lon) — toạ độ trung tâm bản đồ. Center đang dùng là khu vực TP.HCM
        zoom: mức phóng to (12-18 là hợp lý).
        """
        self.screen_width = width
        self.screen_height = height
        self.center = center
        self.zoom = zoom
        self.surface = None

        self.depot_color = 'red'
        self.customer_node_color = 'green'
        self.charge_node_color = 'blue'

    def render_to_surface(self):
        """
        Tạo bản đồ dạng bitmap tiles giống Google Maps và chuyển sang pygame Surface.
        """
        self.m = StaticMap(self.screen_width, self.screen_height, url_template='http://a.tile.openstreetmap.org/{z}/{x}/{y}.png')

        # Render ra ảnh PIL
        image = self.m.render(zoom=self.zoom, center=(self.center[1], self.center[0]))  # lon, lat

        # Chuyển ảnh PIL sang pygame Surface
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        self.surface = pygame.image.load(buffer).convert()

    def get_surface(self):
        if self.surface is None:
            raise ValueError("Chưa render bản đồ.")
        return self.surface

    def render_nodes(self, nodes: list, node_edges: list):
        pass
        # for idx, node in enumerate(nodes):
        #     if idx == 0: # Depot
        #         self.m.add_marker(CircleMarker((), self.depot_color, 15))
        #     else:
        #         self.m.add_marker(CircleMarker((), ''))
