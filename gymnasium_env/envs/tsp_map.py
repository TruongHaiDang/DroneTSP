from staticmap import StaticMap, CircleMarker
import pygame
from io import BytesIO


class TspMap:
    def __init__(self, width=1920, height=1080, center=(10.7769, 106.7009), zoom=15):
        """
        center: tuple(lat, lon) — toạ độ trung tâm bản đồ.
        zoom: mức phóng to (12-18 là hợp lý).
        """
        self.screen_width = width
        self.screen_height = height
        self.center = center
        self.zoom = zoom
        self.surface = None
        self.all_nodes = []

    def render_to_surface(self):
        """
        Tạo bản đồ dạng bitmap tiles giống Google Maps và chuyển sang pygame Surface.
        """
        m = StaticMap(self.screen_width, self.screen_height, url_template='http://a.tile.openstreetmap.org/{z}/{x}/{y}.png')

        # Optionally: thêm marker trung tâm bản đồ
        m.add_marker(CircleMarker((self.center[1], self.center[0]), 'red', 12))

        # Render ra ảnh PIL
        image = m.render(zoom=self.zoom, center=(self.center[1], self.center[0]))  # lon, lat

        # Chuyển ảnh PIL sang pygame Surface
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        self.surface = pygame.image.load(buffer).convert()

    def get_surface(self):
        if self.surface is None:
            raise ValueError("Chưa render bản đồ.")
        return self.surface

    def set_nodes(self, nodes: list):
        self.all_nodes = nodes
