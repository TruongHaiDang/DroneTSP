from staticmap import StaticMap, CircleMarker, Line
import pygame
from io import BytesIO

class TspMap:
    def __init__(self, width=1920, height=1080, center=(10.7769, 106.7009), zoom=15):
        self.screen_width = width
        self.screen_height = height
        self.center = center
        self.zoom = zoom
        self.surface = None
        self.m = None  # StaticMap object

    def begin_render(self):
        self.m = StaticMap(
            self.screen_width, self.screen_height,
            url_template='http://a.tile.openstreetmap.org/{z}/{x}/{y}.png'
        )

    def add_nodes(self, nodes: list):
        for i, node in enumerate(nodes):
            color = 'red' if i == 0 else 'green' if node.node_type.value == 1 else 'blue'
            self.m.add_marker(CircleMarker((node.lon, node.lat), color, 20))

    def add_edges(self, path_indices: list, all_nodes: list):
        """
        path_indices: danh sách index của các node đã đi qua theo thứ tự.
        """
        for i in range(1, len(path_indices)):
            a = all_nodes[path_indices[i - 1]]
            b = all_nodes[path_indices[i]]
            self.m.add_line(Line([(a.lon, a.lat), (b.lon, b.lat)], 'black', 3))

    def commit(self):
        img = self.m.render(zoom=self.zoom, center=(self.center[1], self.center[0]))
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        self.surface = pygame.image.load(buffer).convert()

    def get_surface(self):
        if self.surface is None:
            raise ValueError("Bản đồ chưa được render.")
        return self.surface
