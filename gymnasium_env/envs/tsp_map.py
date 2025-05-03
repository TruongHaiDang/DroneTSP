import osmnx as ox
import matplotlib.pyplot as plt
from io import BytesIO
import pygame

class TspMap:
    """
    Class để tải và vẽ bản đồ OSM lên pygame surface.
    """

    def __init__(self, place_name="Ho Chi Minh City, Vietnam", width=1920, height=1080):
        """
        Khởi tạo với tên địa điểm và kích thước màn hình.
        """
        self.place_name = place_name
        self.screen_width = width
        self.screen_height = height
        self.graph = None
        self.surface = None

    def load_map(self):
        """
        Tải mạng lưới đường phố từ OpenStreetMap sử dụng osmnx.
        """
        # Tải mạng lưới đường phố dành cho xe hơi
        self.graph = ox.graph_from_place(self.place_name, network_type='drive')

    def render_to_surface(self):
        """
        Dùng matplotlib để render bản đồ, sau đó chuyển đổi sang pygame Surface.
        """
        if self.graph is None:
            raise ValueError("Bạn cần gọi load_map() trước khi render.")

        # Vẽ mạng lưới với matplotlib
        fig, ax = ox.plot_graph(
            self.graph, show=False, close=False, figsize=(self.screen_width / 100, self.screen_height / 100)
        )

        # Lưu figure vào buffer
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buffer.seek(0)

        # Load ảnh vào pygame
        self.surface = pygame.image.load(buffer).convert()

    def get_surface(self):
        """
        Trả về pygame Surface đã render, sẵn sàng để blit lên canvas.
        """
        if self.surface is None:
            raise ValueError("Chưa có surface — hãy gọi render_to_surface() trước.")
        return self.surface
