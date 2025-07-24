"""
高多样性导航地图生成器（完整代码）
特性：多类型障碍混合、自适应连通性保障、分层次细节生成
"""
import math
import random
import numpy as np
import cv2
import networkx as nx
import argparse
import time
import cv2
from pathlib import Path
from scipy.spatial import Voronoi, Delaunay
from shapely.geometry import Polygon

class AdvancedMapGenerator:
    def __init__(self, size=1024):
        self.size = size

        self.clutter_config = {
            'node_variety': [(5, 0.2), (6, 0.3), (7, 0.4), (8, 0.3), (9, 0.4), (10, 0.1)],  # (节点数, 凹度概率)
            'size_range': (10, 15),  # 像素单位
            'density_curve': lambda x: 1/(1+np.exp(-0.03*(x-50)))  # 大小分布曲线


        }
        self.obstacle_config = {
            'node_variety': [(5, 0.0), (6, 0.3), (7, 0.4), (8, 0.3), (9, 0.4), (12, 0.3)],  # (节点数, 凹度概率)
            'size_range': (10, 50),  # 像素单位
            'density_curve': lambda x: 1/(1+np.exp(-0.03*(x-50)))  # 大小分布曲线
        }
        self.central_clear_size = 60
        self.wall_config = {
            'min_walls': 2,
            'max_walls': 4,
            'door_width_range': (80, 180),  # 像素单位
            'wall_thickness': 40
        }
        self.connectivity_engine = ConnectivityOptimizer(size)

    def generate_map(self, level = 0, type = "clutter_and_obstacle"):
        """主生成流程"""
        # 阶段1：生成基础障碍层
        print("create obstacle layer")
        obstacle_min_dist = 50
        clutter_min_dist = 70
        if level == 0:
            obstacle_min_dist = 110
            clutter_min_dist = 70
        elif level == 1:
            obstacle_min_dist = 100
            clutter_min_dist = 60

        elif level == 2:
            obstacle_min_dist = 90
            clutter_min_dist = 50

        elif level == 3:
            obstacle_min_dist = 80
            clutter_min_dist = 40

        elif level == 4:
            obstacle_min_dist = 70
            clutter_min_dist = 30

        elif level == 5:
            obstacle_min_dist = 60
            clutter_min_dist = 20





        base = np.ones((self.size, self.size), np.uint8) * 255
        # base = self.add_structured_obstacles(base)
        if type == "obstacle":
            base = self.create_obstacle_layer(base, min_dist=obstacle_min_dist)
        elif type == "clutter":
        # base = self.add_random_obstacles(base)
            base = self.create_clutter_layer(base, min_dist=clutter_min_dist)
        elif type == "obstacle_and_clutter":
            base = self.create_obstacle_layer(base, min_dist=obstacle_min_dist)
            base = self.create_clutter_layer(base, min_dist=clutter_min_dist)
        elif type == "tunnel":
            print("create tunnel layer")
            base = self.create_tunnel_layer(base)
        elif type == "empty":
            base = self.create_empty_layer(base)
            # Tunnel maps have a predefined structure and don't need the standard
            # connectivity optimization, which would create shortcuts through walls.
        elif type == "test":
            base = self.create_test_layer(base)


        # base = self.add_central_clear_area(base)
        print("morphological enhance")
        # 阶段2：形态学细节增强
        # base = self.morphological_detail(base)
        

        print("connectivity optimization")
        # 阶段3：多层次连通优化
        return base
        # return self.connectivity_engine.optimize(base)
        # return base

    def create_arc_tube_obstacle(self, center, base_radius, start_angle, end_angle, thickness, irregularity=0.15, segments=30):
        """
        Generates a single, irregular arc-shaped tube obstacle.
        :param center: Tuple (x, y) for the center of the arc.
        :param base_radius: The average distance from the center to the midline of the arc.
        :param start_angle: The starting angle of the arc in degrees.
        :param end_angle: The ending angle of the arc in degrees.
        :param thickness: The average thickness of the arc tube.
        :param irregularity: A factor to control the "wobbliness" of the arc's shape.
        :param segments: The number of line segments to use to approximate the curve.
        :return: A numpy array of points defining the polygon of the obstacle.
        """
        outer_arc = []
        inner_arc = []

        # Generate points along the arc
        for i in range(segments + 1):
            # Interpolate angle and convert to radians
            angle_rad = math.radians(start_angle + (end_angle - start_angle) * i / segments)

            # Apply irregularity to radius and thickness for a more organic shape
            current_radius = base_radius * (1 + random.uniform(-irregularity, irregularity) * 0.5)
            current_thickness = thickness * (1 + random.uniform(-irregularity, irregularity))

            # Calculate points for the outer and inner edges of the arc
            outer_x = center[0] + (current_radius + current_thickness / 2) * math.cos(angle_rad)
            outer_y = center[1] + (current_radius + current_thickness / 2) * math.sin(angle_rad)
            outer_arc.append([int(outer_x), int(outer_y)])

            inner_x = center[0] + (current_radius - current_thickness / 2) * math.cos(angle_rad)
            inner_y = center[1] + (current_radius - current_thickness / 2) * math.sin(angle_rad)
            inner_arc.append([int(inner_x), int(inner_y)])
        
        # Combine the outer and inner arc points to form a closed polygon.
        # The inner arc points are added in reverse to create a continuous path for the polygon boundary.
        polygon_points = np.array(outer_arc + inner_arc[::-1], dtype=np.int32)
        return polygon_points

    def create_empty_layer(self, base):
        base.fill(255) 
        return base

    def create_test_layer(self, base):
        base.fill(255)  # Start with a completely white (free space) map

        initial = (self.size*0.7, self.size*0.5)
        nodes, concave_prob = self.clutter_config['node_variety'][
                np.random.choice(len(self.clutter_config['node_variety']), 
                               p=[0.2, 0.2, 0.1, 0.1, 0.2, 0.2])]
            
        # 生成随机尺寸（带密度控制）
        size = self.sample_clutter_size()
            
            # 生成多边形顶点
        poly = self.generate_convex_poly(initial, nodes, size)
            
        cv2.fillPoly(base, [poly], 0)
        base = self.add_central_half_clear_area(base)
        return base
    
    def create_tunnel_layer(self, base):
        """
        Creates a map with several arc-tube obstacles surrounding a central area,
        forming tunnels between them, with a controlled tunnel width.
        """
        base.fill(255)  # Start with a completely white (free space) map
        center = (self.size // 2, self.size // 2)

        # --- Configuration for the arc-tube tunnel style ---
        num_arcs = random.randint(3, 7)
        mean_radius = self.size * random.uniform(0.25, 0.35)
        thickness = self.size * random.uniform(0.03, 0.15)
        
        # --- New logic to control tunnel width ---
        # Define the desired tunnel width (gap) in degrees from config
        # target_gap_angle = random.uniform(*self.tunnel_config['width_range_deg'])
        target_gap_angle = random.uniform(8,13)
        
        total_gap_angle = num_arcs * target_gap_angle

        # Ensure total gap is not excessive, leaving space for arcs
        if total_gap_angle >= 300:
            total_gap_angle = 300 # Cap the total gap angle

        # Calculate the average arc length based on the required gaps
        total_arc_angle = 360 - total_gap_angle
        arc_length_mean = total_arc_angle / num_arcs if num_arcs > 0 else 0
        # --- End of new logic ---

        current_angle = random.uniform(0, 360)

        for _ in range(num_arcs):
            # Randomize parameters for this specific arc for variety
            # Use less variance to maintain tunnel width more consistently
            arc_length = arc_length_mean + random.uniform(-10, 3)
            start_angle = current_angle
            end_angle = start_angle + arc_length

            # Generate the polygon for the current arc-shaped obstacle
            arc_poly = self.create_arc_tube_obstacle(
                center=center,
                base_radius=mean_radius * random.uniform(0.7, 1.2),
                start_angle=start_angle,
                end_angle=end_angle,
                thickness=thickness * random.uniform(0.4, 1.9),
                irregularity=0.2, # Controls how "wobbly" the arcs are
                segments=10       # More segments for smoother curves
            )
            
            # Draw the obstacle on the map
            cv2.fillPoly(base, [arc_poly], 0) 

            # Update the current_angle for the next arc, leaving a controlled gap
            gap_angle = target_gap_angle + random.uniform(-1, 2)
            current_angle = end_angle + gap_angle
            
        return base
    def add_structured_obstacles(self, base):
            """生成带门的墙体系统"""
            num_walls = random.randint(self.wall_config['min_walls'], 
                                    self.wall_config['max_walls'])
            
            for _ in range(num_walls):
                # 随机选择墙体方向（0:水平，1:垂直）
                orientation = random.choice([0, 1])
                
                # 生成墙体参数
                door_width = random.randint(*self.wall_config['door_width_range'])
                wall_thickness = self.wall_config['wall_thickness']
                
                # 绘制带门的墙体
                if orientation == 0:  # 水平墙
                    y = random.randint(self.size//4, 3*self.size//4)
                    split_pos = random.randint(self.size//4, 3*self.size//4)
                    
                    # 绘制墙的左右部分
                    cv2.rectangle(base, (0, y), (split_pos - door_width//2, y + wall_thickness), 0, -1)
                    cv2.rectangle(base, (split_pos + door_width//2, y), (self.size, y + wall_thickness), 0, -1)
                else:  # 垂直墙
                    x = random.randint(self.size//4, 3*self.size//4)
                    split_pos = random.randint(self.size//4, 3*self.size//4)
                    
                    cv2.rectangle(base, (x, 0), (x + wall_thickness, split_pos - door_width//2), 0, -1)
                    cv2.rectangle(base, (x, split_pos + door_width//2), (x + wall_thickness, self.size), 0, -1)
            
            return base
    def generate_random_polygon(self, center, vertices, radius, concavity=0.5):
        """生成带凹度的随机多边形"""
        angles = sorted([random.uniform(0, 2*math.pi) for _ in range(vertices)])
        points = []
        
        # 生成基础顶点
        for i, angle in enumerate(angles):
            var_radius = radius * (1 + random.uniform(-0.2, 0.2))
            x = center[0] + var_radius * math.cos(angle)
            y = center[1] + var_radius * math.sin(angle)
            points.append((x, y))
        
        # 引入凹度
        if random.random() < concavity:
            concave_idx = random.randint(0, vertices-1)
            next_idx = (concave_idx + 1) % vertices
            prev_idx = (concave_idx - 1) % vertices
            
            # 计算凹陷点
            mid_angle = (angles[prev_idx] + angles[next_idx]) / 2
            depth = radius * 0.4
            concave_pt = (
                center[0] + depth * math.cos(mid_angle),
                center[1] + depth * math.sin(mid_angle)
            )
            points[concave_idx] = concave_pt
        
        return np.array(points, dtype=np.int32)
    
    def detect_door_areas(self, img):
        """使用形态学操作检测门口区域"""
        # 转换为二值图像
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # 识别水平门
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,1))
        hor_doors = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # 识别垂直门
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,30))
        ver_doors = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # 合并门区域
        door_mask = cv2.bitwise_or(hor_doors, ver_doors)
        return door_mask

    def is_near_door(self, point, door_mask, radius=50):
        """检查点是否在门口附近"""
        x, y = int(point[0]), int(point[1])
        y1, y2 = max(0, y-radius), min(self.size, y+radius)
        x1, x2 = max(0, x-radius), min(self.size, x+radius)
        roi = door_mask[y1:y2, x1:x2]
        return np.any(roi == 255)

    def add_random_obstacles(self, base):
        """添加自然分布的随机障碍物"""
        # 生成泊松采样点（避开门口区域）
        points = self.poisson_disk_sampling(min_dist=60)
        
        # 识别门口区域
        door_mask = self.detect_door_areas(base)
        
        # 过滤靠近门口的采样点
        valid_points = [pt for pt in points if not self.is_near_door(pt, door_mask)]
        
        # 绘制多边形障碍
        for (x,y) in valid_points:
            vertices = random.randint(5, 8)
            radius = random.randint(20, 50)
            poly = self.generate_random_polygon((x,y), vertices, radius)
            cv2.fillPoly(base, [poly], 0)
        
        return base

    def find_door_centers(self, img):
        """精确定位门口中心"""
        door_mask = self.detect_door_areas(img)
        contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:  # 过滤噪声
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
        return centers

    def poisson_disk_sampling(self, min_dist, max_attempts=30, door_mask=None):
        """改进的泊松采样，避开门口区域"""
        cell_size = min_dist / math.sqrt(2)
        grid_size = int(math.ceil(self.size / cell_size))
        grid = [[None for _ in range(grid_size)] for __ in range(grid_size)]
        process_list = []
        samples = []

        def get_grid_coords(pt):
            return int(pt[0]//cell_size), int(pt[1]//cell_size)

        def is_valid_sample(pt):
            if door_mask and door_mask[int(pt[1]), int(pt[0])]:
                return False  # 避开门口区域
            
            gx, gy = get_grid_coords(pt)
            for x in range(max(gx-2, 0), min(gx+3, grid_size)):
                for y in range(max(gy-2, 0), min(gy+3, grid_size)):
                    s = grid[x][y]
                    if s is not None:
                        distance = math.hypot(s[0]-pt[0], s[1]-pt[1])
                        if distance < min_dist:
                            return False
            return True

        # 初始点
        initial = (self.size*random.random(), self.size*random.random())
        while not is_valid_sample(initial):
            initial = (self.size*random.random(), self.size*random.random())
        
        process_list.append(initial)
        samples.append(initial)
        gx, gy = get_grid_coords(initial)
        grid[gx][gy] = initial

        while process_list:
            idx = random.randint(0, len(process_list)-1)
            point = process_list[idx]
            
            for _ in range(max_attempts):
                angle = 2 * math.pi * random.random()
                radius = min_dist * (1 + random.random())
                new_pt = (point[0] + radius*math.cos(angle),
                          point[1] + radius*math.sin(angle))
                
                if 0 <= new_pt[0] < self.size and 0 <= new_pt[1] < self.size:
                    if is_valid_sample(new_pt):
                        process_list.append(new_pt)
                        samples.append(new_pt)
                        gx, gy = get_grid_coords(new_pt)
                        grid[gx][gy] = new_pt
                        break
            else:
                del process_list[idx]
                
        return samples

    def _add_point(self, point, grid, cell_size, points, active):
        """将有效点添加到数据结构"""
        grid_col = int(point[0] / cell_size)
        grid_row = int(point[1] / cell_size)
        grid[grid_col, grid_row] = len(points)
        points.append(point)
        active.append(point)

    def _is_valid_point(self, point, grid, cell_size, min_dist, points):
        """验证点是否符合泊松采样条件"""
        # 边界检查
        if not (0 <= point[0] < self.size and 0 <= point[1] < self.size):
            return False

        # 网格索引计算
        col = int(point[0] / cell_size)
        row = int(point[1] / cell_size)
        start_col = max(0, col - 2)
        end_col = min(grid.shape[0], col + 3)
        start_row = max(0, row - 2)
        end_row = min(grid.shape[1], row + 3)

        # 邻近网格检查
        for c in range(start_col, end_col):
            for r in range(start_row, end_row):
                if grid[c, r] != -1:
                    existing_point = points[grid[c, r]]
                    distance = np.hypot(point[0] - existing_point[0],
                                       point[1] - existing_point[1])
                    if distance < min_dist:
                        return False
        return True
    
    def create_clutter_layer(self, base, min_dist=70):
        """创建多样化障碍层"""
        points = self.poisson_disk_sampling(min_dist=min_dist)
        
        for center in points:
            # 随机选择障碍类型
            nodes, concave_prob = self.clutter_config['node_variety'][
                np.random.choice(len(self.clutter_config['node_variety']), 
                               p=[0.2, 0.2, 0.1, 0.1, 0.2, 0.2])]
            
            # 生成随机尺寸（带密度控制）
            size = self.sample_clutter_size()
            
            # 生成多边形顶点
            if np.random.rand() < concave_prob:
                poly = self.generate_concave_poly(center, nodes, size)
            else:
                poly = self.generate_convex_poly(center, nodes, size)
            
            cv2.fillPoly(base, [poly], 0)
        
        return base

    def create_obstacle_layer(self, base, min_dist=50):
        """创建多样化障碍层"""
        points = self.poisson_disk_sampling(min_dist=min_dist)
        
        for center in points:
            # 随机选择障碍类型
            nodes, concave_prob = self.obstacle_config['node_variety'][
                np.random.choice(len(self.obstacle_config['node_variety']), 
                               p=[0.2, 0.2, 0.1, 0.1, 0.2, 0.2])]
            
            # 生成随机尺寸（带密度控制）
            size = self.sample_size()
            
            # 生成多边形顶点
            if np.random.rand() < concave_prob:
                poly = self.generate_concave_poly(center, nodes, size)
            else:
                poly = self.generate_convex_poly(center, nodes, size)
            
            cv2.fillPoly(base, [poly], 0)
        
        return base

    def sample_clutter_size(self):
        """基于概率密度曲线的尺寸采样"""
        x = np.arange(*self.clutter_config['size_range'])
        pdf = self.clutter_config['density_curve'](x)
        pdf /= pdf.sum()
        return np.random.choice(x, p=pdf)

    def sample_size(self):
        """基于概率密度曲线的尺寸采样"""
        x = np.arange(*self.obstacle_config['size_range'])
        pdf = self.obstacle_config['density_curve'](x)
        pdf /= pdf.sum()
        return np.random.choice(x, p=pdf)

    def generate_concave_poly(self, center, nodes, size, 
                         max_concave_ratio=0.6, 
                         max_depth_ratio=0.7):
        """增强版凹多边形生成器"""
        # 参数校验
        assert 3 <= nodes <= 12, "节点数应在3-12之间"
        assert 0.3 <= max_concave_ratio <= 0.8, "凹点比例需在30%-80%"
        
        # 生成基础顶点（带角度噪声）
        base_angles = np.sort(np.random.normal(
            np.linspace(0, 2*np.pi, nodes),
            scale=np.pi/(nodes*2)
        )) % (2*np.pi)
        
        # 生成随机半径（带波动性）
        radii = np.random.weibull(1.5, nodes) * size * 0.6 + size*0.4
        radii = np.clip(radii, size*0.3, size*2.0)
        
        # 凹点生成策略
        concave_num = min(
            np.random.randint(3, 5),  # 强制至少3个凹点
            int(nodes * max_concave_ratio)
        )
        concave_indices = self._select_concave_positions(nodes, concave_num)
        
        # 深度凹陷生成
        for i in concave_indices:
            # 计算动态凹陷参数
            depth_factor = np.random.uniform(0.2, max_depth_ratio)  # 凹陷深度可达80%
            angle_shift_range = np.pi/(nodes*2)  # 角度调整范围
            
            # 获取相邻点索引
            prev_idx = (i-1) % nodes
            next_idx = (i+1) % nodes
            
            # 计算凹陷锚点
            anchor_angle = (base_angles[prev_idx] + base_angles[next_idx])/2
            base_angles[i] = anchor_angle + np.random.uniform(-angle_shift_range, angle_shift_range)
            
            # 应用深度凹陷
            radii[i] *= depth_factor
            
            # 添加次级凹陷（30%概率）
            if np.random.rand() < 0.3:
                secondary_idx = (i + np.random.choice([-2,2])) % nodes
                radii[secondary_idx] *= np.random.uniform(0.5, 0.8)
        
        # 顶点排序和形状验证
        sorted_indices = np.argsort(base_angles)
        sorted_angles = base_angles[sorted_indices]
        sorted_radii = radii[sorted_indices]
        
        poly = self._create_polygon(center, sorted_angles, sorted_radii)
        return self._validate_concave_shape(poly)

    def _select_concave_positions(self, nodes, concave_num, min_gap=2):
        """智能选择凹点位置，避免相邻"""
        positions = []
        candidates = list(range(nodes))
        
        while len(positions) < concave_num and candidates:
            pos = np.random.choice(candidates)
            positions.append(pos)
            # 移除邻近位置
            candidates = [x for x in candidates 
                        if abs(x - pos) >= min_gap 
                        and abs(x - pos) % nodes >= min_gap]
        
        return positions[:concave_num]
    
    def add_central_clear_area(self, base):
        """添加中央方形空地"""
        x_start = (self.size - self.central_clear_size) // 2
        y_start = (self.size - self.central_clear_size) // 2
        x_end = x_start + self.central_clear_size
        y_end = y_start + self.central_clear_size
        
        cv2.rectangle(base, 
                     (x_start, y_start),
                     (x_end, y_end),
                     255, -1)  # 白色填充
        return base
    
    def add_central_half_clear_area(self, base):
        """添加中央方形空地"""
        x_start = (self.size - self.central_clear_size) // 2
        y_start = (self.size - self.central_clear_size) // 2
        x_end = x_start + self.central_clear_size // 2
        y_end = y_start + self.central_clear_size
        
        cv2.rectangle(base, 
                     (x_start, y_start),
                     (x_end, y_end),
                     255, -1)  # 白色填充
        return base

    def _validate_concave_shape(self, poly):
        """形状有效性验证"""
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        
        shapely_poly = Polygon(poly)
        if not shapely_poly.is_valid:
            # 自动修复无效多边形
            repaired = make_valid(shapely_poly)
            if repaired.geom_type == 'Polygon':
                return np.array(repaired.exterior.coords[:-1], dtype=np.int32)
        return poly

    def generate_convex_poly(self, center, nodes, base_radius, 
                        angle_perturb=0.3, radius_variance=0.4):
        """
        生成自然风格的凸多边形
        :param center: 中心点坐标 (x,y)
        :param nodes: 顶点数
        :param base_radius: 基础半径（像素）
        :param angle_perturb: 最大角度扰动比例（0-1）
        :param radius_variance: 半径随机变化比例（0-1）
        :return: 凸多边形顶点坐标数组
        """
        # 生成基础角度（带扰动）
        base_angles = np.linspace(0, 2*np.pi, nodes, endpoint=False)
        angle_perturb_max = 2*np.pi/nodes * angle_perturb  # 计算最大允许扰动
        
        # 添加角度扰动并保持递增顺序
        perturbed_angles = []
        current_angle = 0
        for i in range(nodes):
            # 保证每个角度区间不超过 (2π/nodes)
            max_next = current_angle + (2*np.pi/nodes)*(1 + angle_perturb)
            next_angle = current_angle + (2*np.pi/nodes)*np.random.uniform(1-angle_perturb, 1)
            next_angle = min(next_angle, max_next)
            perturbed_angles.append(next_angle)
            current_angle = next_angle
        
        # 归一化到0-2π范围
        perturbed_angles = np.array(perturbed_angles) % (2*np.pi)
        perturbed_angles.sort()  # 确保角度严格递增
        
        # 生成半径（带随机变化）
        radii = base_radius * np.random.uniform(1 - radius_variance, 
                                            1 + radius_variance, 
                                            size=nodes)
        
        # 生成顶点坐标
        points = []
        for angle, r in zip(perturbed_angles, radii):
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            points.append((x, y))
        
        # 转换为整数坐标并验证凸性
        poly = np.array(points, dtype=np.int32)
        
        # 凸性验证（确保至少99%的点在凸包内）
        hull = cv2.convexHull(poly)
        if len(hull) < nodes*0.95:  # 允许少量顶点被凸包优化
            return self.generate_convex_poly(center, nodes, base_radius)  # 递归重新生成
        
        return poly

    def _create_polygon(self, center, angles, radii):
        """通用多边形生成方法"""
        points = []
        for a, r in zip(angles, radii):
            x = center[0] + r * np.cos(a)
            y = center[1] + r * np.sin(a)
            points.append([x, y])
        return np.array(points, np.int32)

    def morphological_detail(self, base):
        """多尺度形态学增强"""
        # 第一阶段：粗粒度膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        temp = cv2.dilate(base, kernel)
        
        # 第二阶段：随机腐蚀
        for _ in range(3):
            rand_kernel = np.random.rand(3,3) > 0.9
            temp = cv2.erode(temp, rand_kernel.astype(np.uint8))
        
        # 第三阶段：细粒度膨胀
        return cv2.dilate(temp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

class ConnectivityOptimizer:
    """增强型连通性优化引擎"""
    def __init__(self, size):
        self.size = size
        self.channel_min = 30  # 最小通道宽度
    
    def optimize(self, img):
        """三级连通优化"""
        # 预处理
        print("preprocess")
        img = self._preprocess(img)
        

        print("build region graph")
        # 连通域分析
        graph = self._build_region_graph(img)
        

        print("global repair")
        # 全局连通修复
        return self._global_repair(img, graph)

    def _preprocess(self, img):
        """连通性预处理"""
        # 创建边界安全区
        cv2.rectangle(img, (0,0), (self.size, self.size), 0, self.channel_min*2)
        return cv2.GaussianBlur(img, (5,5), 0) > 0.5

    def _build_region_graph(self, img):
        """构建区域拓扑图"""
        _, labels = cv2.connectedComponents(img.astype(np.uint8))
        regions = np.unique(labels)
        
        graph = nx.Graph()
        for r in regions:
            if r == 0: continue
            mask = (labels == r).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            graph.add_node(r, contour=contours[0], area=cv2.contourArea(contours[0]))
        
        return graph

    def _global_repair(self, img, graph):
        """全局连通性修复"""
        tried_num = 0
        while nx.number_connected_components(graph) > 5:
            print("num of connected com:{}".format(nx.number_connected_components(graph)))
            if tried_num > 30:
                break
            components = list(nx.connected_components(graph))
            main = max(components, key=lambda x: sum(graph.nodes[n]['area'] for n in x))
            
            for comp in components:
                if comp != main:
                    path = self._find_optimal_path(graph, main, comp)
                    self._carve_path(img, path)
            tried_num += 1
        print("finish global repair")
        return img * 255

    def _find_optimal_path(self, graph, main, target):
        """寻找最优连接路径"""
        main_points = [self._get_contour_center(graph.nodes[n]['contour']) for n in main]
        target_points = [self._get_contour_center(graph.nodes[n]['contour']) for n in target]
        
        # 寻找最近点对
        min_dist = np.inf
        optimal_pair = None
        for p1 in main_points:
            for p2 in target_points:
                dist = np.linalg.norm(np.array(p1)-np.array(p2))
                if dist < min_dist:
                    min_dist = dist
                    optimal_pair = (p1, p2)
        
        return optimal_pair

    def _carve_path(self, img, path):
        """雕刻连通路径（修复数据类型问题）"""
        # 步骤1：确保数据类型为uint8
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        """雕刻连通路径"""
        p1, p2 = path
        # print("carve path")
        cv2.line(img, tuple(map(int, p1)), tuple(map(int, p2)), 0, self.channel_min)
        
        # 路径形态学优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.channel_min, self.channel_min))
        img = cv2.dilate(img, kernel)
        return cv2.erode(img, kernel)

    def _get_contour_center(self, contour):
        """获取轮廓质心"""
        M = cv2.moments(contour)
        return (M['m10']/(M['m00']+1e-5), M['m01']/(M['m00']+1e-5))
    


    


def main():
    """命令行接口主函数"""
    parser = argparse.ArgumentParser(description='生成高连通性导航地图')
    parser.add_argument('--num', type=int, default=5, 
                      help='生成地图数量，默认5')
    parser.add_argument('--size', type=int, default=256,
                      choices=[128, 256, 512, 768, 1024, 2048],
                      help='地图尺寸（像素），默认512')
    parser.add_argument('--output', type=str, default='./maps',
                      help='输出目录路径，默认./maps')
    parser.add_argument('--version', action='version', version='v2.1.0')
    parser.add_argument('--type', type=str, default='obstacle_and_clutter')
    parser.add_argument('--level', type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--path', type=int, default=0, help='0-no path generated, 1 - generate paths')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化生成器
    print(f"初始化{args.size}x{args.size}地图生成器...")
    generator = AdvancedMapGenerator(size=args.size)
    
    # 生成进度跟踪
    success_count = 0
    start_time = time.time()
    
    while success_count < args.num:
        try:
            # 生成单张地图
            map_id = success_count + 1
            print(f"\n生成地图 #{map_id}/{args.num}")
            
            # 执行生成流程
            map_data = generator.generate_map(level = args.level, type = args.type)
            
            # 质量验证
            if validate_map(map_data):
            # if True:
                filename = output_dir / f"map_{map_id}_{args.size}px_{args.type}.png"
                print("before saving")
                cv2.imwrite(str(filename), map_data)
                print(f"已保存：{filename.name}")
                success_count += 1

                map_name = f"map_{map_id}_{args.size}px"
                agent_radius = 0.5  # Set appropriate agent radius
                
                
                


            else:
                print("质量检查失败，重新生成...")
                
        except Exception as e:
            print(f"生成错误：{str(e)}")
            continue

    # 性能统计
    total_time = time.time() - start_time
    print(f"\n生成完成！平均耗时：{total_time/args.num:.2f}s/张")
    print(f"输出目录：{output_dir.resolve()}")


def validate_map(img):
    """Enhanced map quality validation function with connectivity checks"""
    # Ensure image is in correct format for processing
    if img.dtype == bool:
        binary = img
    else:
        binary = img > 127  # Convert to binary (True for free space, False for obstacles)
    
    # Check obstacle ratio
    # obstacle_ratio = np.mean(img < 127) 
    # if not 0.2 < obstacle_ratio < 0.7:
    #     print(f"障碍物比例异常：{obstacle_ratio:.2f}")
    #     return False
    
    # Define the central clear area
    height, width = binary.shape
    central_clear_size = 60  # Should match the value in add_central_clear_area
    center_y, center_x = height // 2, width // 2
    
    # Make sure center point is in free space
    if not binary[center_y, center_x]:
        print("中央区域被障碍物覆盖")
        return False
    
    # Select 3 random points outside the central region but in free space
    random_positions = []
    attempts = 0
    while len(random_positions) < 3 and attempts < 100:
        end_y = random.randint(0, height - 1)
        end_x = random.randint(0, width - 1)
        
        # Check if point is outside central area and is free space
        outside_central = (abs(end_y - center_y) > central_clear_size//2 or 
                          abs(end_x - center_x) > central_clear_size//2)
        if outside_central and binary[end_y, end_x]:
            random_positions.append((end_y, end_x))
        
        attempts += 1
    
    # If we couldn't find enough random positions
    if len(random_positions) < 3:
        print(f"找不到足够的可访问随机位置，仅找到{len(random_positions)}个")
        return False
    
    # Check if there is a path to each of the random positions
    for i, (end_y, end_x) in enumerate(random_positions):
        if not has_path(binary, (center_y, center_x), (end_y, end_x)):
            print(f"中央区域无法到达随机位置{i+1}: ({end_x}, {end_y})")
            return False
    
    return True

def has_path(grid, start, end):
    """
    使用广度优先搜索检查从起点到终点的路径是否存在
    
    参数:
        grid: 二值网格，True表示可通行区域，False表示障碍物
        start: (row, col) 起始位置
        end: (row, col) 目标位置
    
    返回:
        如果存在路径则返回True，否则返回False
    """
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    visited[start] = True
    queue = [start]
    
    # 定义可能的移动方向: 上、下、左、右
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        curr_row, curr_col = queue.pop(0)
        
        if (curr_row, curr_col) == end:
            return True
        
        for dr, dc in moves:
            new_row, new_col = curr_row + dr, curr_col + dc
            
            if (0 <= new_row < height and 0 <= new_col < width and 
                grid[new_row, new_col] and not visited[new_row, new_col]):
                visited[new_row, new_col] = True
                queue.append((new_row, new_col))
    
    return False




# def validate_map(img):
# #     """地图质量验证"""
# #     # 连通性验证
# #     print("validate map")
# #     _, labels = cv2.connectedComponents(255 - img)
# #     if np.max(labels) > 1:
# #         print("连通性验证失败：存在多个独立区域")
# #         return False
    
# #     # 障碍物覆盖率验证
#     obstacle_ratio = np.mean(img < 127) 
#     if not 0.2 < obstacle_ratio < 0.6:
#         print(f"障碍物比例异常：{obstacle_ratio:.2f}")
#         return False
    
#     return True

# def validate_map(img):
#     """改进后的地图质量验证函数"""
#     # 强制类型转换确保输入为8位无符号整型
#     if img.dtype == bool:
#         img_uint8 = img.astype(np.uint8) * 255
#     else:
#         img_uint8 = cv2.convertScaleAbs(img)

#     # 连通性验证（使用二值化处理）
#     _, binary = cv2.threshold(img_uint8, 127, 255, cv2.THRESH_BINARY)
#     num_labels, labels = cv2.connectedComponents(binary)
    
#     if num_labels > 3:  # 包含背景至少有两个区域
#         print(f"连通性验证失败：检测到{num_labels-1}个独立障碍区域")
#         return False
    
#     # 障碍物覆盖率验证（匹配参考图像风格）
#     obstacle_ratio = np.sum(binary == 0) / binary.size
#     if not 0.15 < obstacle_ratio < 0.55:
#         print(f"障碍物比例超出标准范围（当前：{obstacle_ratio:.2f}）")
#         return False
    
#     # 通道宽度验证（基于参考图像中的最小通道）
#     dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
#     min_channel = np.max(dist_transform)
#     if min_channel < 20:  # 最小通道宽度像素阈值
#         print(f"通道宽度不足（最小：{min_channel:.1f}px）")
#         return False
    
#     return True


if __name__ == "__main__":
    main()



    #python3 map_generator.py --num 20 --output ../train_maps
