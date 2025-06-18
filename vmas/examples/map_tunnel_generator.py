"""
高多样性隧道导航地图生成器（完整代码）
特性：多类型障碍混合、自适应连通性保障、分层次细节生成、隧道特征
"""
import math
import random
import numpy as np
import cv2
import networkx as nx
import argparse
import time
from pathlib import Path
# from scipy.spatial import Voronoi, Delaunay # Not used in provided base
# from shapely.geometry import Polygon # Used internally in some methods

# It seems shapely.geometry.Polygon is used directly, ensure it's available
# If not, functions like _validate_concave_shape might fail.
# For now, assuming it's available as per the original script.
try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
except ImportError:
    print("Warning: Shapely library not found. Some polygon validation features might not work.")
    Polygon = None
    make_valid = None


class TunnelMapGenerator:
    def __init__(self, size=1024,
                 tunnel_start_x_range=(0.6, 0.8),  # As a fraction of map size
                 tunnel_height_range=(0.1, 0.25) # As a fraction of map size
                 ):
        self.size = size
        self.tunnel_start_x_config_range = tunnel_start_x_range
        self.tunnel_height_config_range = tunnel_height_range

        # Instance variables to store actual tunnel params for validation
        self.current_tunnel_start_x = None
        self.current_tunnel_height = None

        self.clutter_config = {
            'node_variety': [(5, 0.2), (6, 0.3), (7, 0.4), (8, 0.3), (9, 0.4), (10, 0.1)],
            'size_range': (max(5, int(size * 0.005)), max(10, int(size * 0.01))), # Relative size
            'density_curve': lambda x: 1/(1+np.exp(-0.03*(x- (self.clutter_config['size_range'][0] + self.clutter_config['size_range'][1])/2 )))
        }
        self.obstacle_config = {
            'node_variety': [(5, 0.0), (6, 0.3), (7, 0.4), (8, 0.3), (9, 0.4), (12, 0.3)],
            'size_range': (max(10, int(size*0.01)), max(50, int(size*0.05))), # Relative size
            'density_curve': lambda x: 1/(1+np.exp(-0.03*(x- (self.obstacle_config['size_range'][0] + self.obstacle_config['size_range'][1])/2 )))
        }
        self.central_clear_size = int(size * 0.06) # Relative size e.g. 60 for 1024
        self.wall_config = { # Not directly used for tunnel, but kept from base
            'min_walls': 2,
            'max_walls': 4,
            'door_width_range': (int(size*0.08), int(size*0.18)),
            'wall_thickness': int(size*0.04)
        }
        self.connectivity_engine = ConnectivityOptimizer(size, channel_min_factor=0.03) # e.g. 30 for 1024

    def generate_map(self, level = 0, type = "obstacle_and_clutter"):
        """主生成流程"""
        base = np.ones((self.size, self.size), np.uint8) * 255

        obstacle_min_dist_base = int(self.size * 0.1)  # e.g. 100 for 1024
        clutter_min_dist_base = int(self.size * 0.07) # e.g. 70 for 1024

        # Adjust distances based on level (example scaling)
        level_factor = 1.0 - (level * 0.1) # Decreases distance for higher levels
        obstacle_min_dist = int(obstacle_min_dist_base * level_factor)
        clutter_min_dist = int(clutter_min_dist_base * level_factor)
        
        # Ensure min_dist is not too small
        obstacle_min_dist = max(int(self.size*0.02), obstacle_min_dist) # Min 2% of size
        clutter_min_dist = max(int(self.size*0.01), clutter_min_dist)   # Min 1% of size


        print("Creating obstacle/clutter layer...")
        if type == "obstacle":
            base = self.create_obstacle_layer(base, min_dist=obstacle_min_dist)
        elif type == "clutter":
            base = self.create_clutter_layer(base, min_dist=clutter_min_dist)
        elif type == "obstacle_and_clutter": # Tunnel maps also start with obstacles
            base = self.create_obstacle_layer(base, min_dist=obstacle_min_dist)
            base = self.create_clutter_layer(base, min_dist=clutter_min_dist)
        elif type == "tunnel":
             base = self.add_tunnel_feature(base)

        print("Applying morphological detail enhancement...")
        base = self.morphological_detail(base)
        
        

        print("Optimizing connectivity...")
        optimized_map = self.connectivity_engine.optimize(base.copy()) # Pass a copy
        
        return optimized_map

    def add_tunnel_feature(self, base_map):
        """Adds a horizontal tunnel to the right half of the map."""
        # Randomize tunnel parameters
        start_x_frac = random.uniform(self.tunnel_start_x_config_range[0], self.tunnel_start_x_config_range[1])
        self.current_tunnel_start_x = int(self.size * start_x_frac)
        
        # Ensure tunnel_start_x is in the right half and gives space for the tunnel
        min_start_x = self.size // 2 + 1
        self.current_tunnel_start_x = max(self.current_tunnel_start_x, min_start_x)
        self.current_tunnel_start_x = min(self.current_tunnel_start_x, self.size - int(self.size * 0.1)) # Ensure some length for tunnel

        height_frac = random.uniform(self.tunnel_height_config_range[0], self.tunnel_height_config_range[1])
        self.current_tunnel_height = int(self.size * height_frac)
        self.current_tunnel_height = max(10, self.current_tunnel_height) # Min height 10px

        print(f"  Tunnel params: start_x={self.current_tunnel_start_x}, height={self.current_tunnel_height}")

        # Calculate tunnel's Y-coordinates (centered)
        tunnel_y_start = (self.size - self.current_tunnel_height) // 2
        tunnel_y_end = tunnel_y_start + self.current_tunnel_height

        # Fill the area to the right of tunnel_start_x with obstacles (0)
        base_map[:, self.current_tunnel_start_x:] = 0

        # Carve out the tunnel (free space = 255)
        base_map[tunnel_y_start:tunnel_y_end, self.current_tunnel_start_x:] = 255
        
        return base_map

    def add_central_clear_area(self, base):
        """添加中央方形空地"""
        x_start = (self.size - self.central_clear_size) // 2
        y_start = (self.size - self.central_clear_size) // 2
        x_end = x_start + self.central_clear_size
        y_end = y_start + self.central_clear_size
        
        cv2.rectangle(base, 
                     (x_start, y_start),
                     (x_end, y_end),
                     255, -1) 
        return base

    def poisson_disk_sampling(self, min_dist, max_attempts=30):
        cell_size = min_dist / math.sqrt(2)
        grid_w = int(math.ceil(self.size / cell_size))
        grid_h = int(math.ceil(self.size / cell_size))
        grid = [[None for _ in range(grid_w)] for _ in range(grid_h)]
        
        process_list = []
        samples = []

        def get_grid_coords(pt):
            return int(pt[0] / cell_size), int(pt[1] / cell_size)

        def is_valid(pt):
            if not (0 <= pt[0] < self.size and 0 <= pt[1] < self.size):
                return False
            
            col, row = get_grid_coords(pt)
            
            # Check neighborhood
            for r_offset in range(-2, 3):
                for c_offset in range(-2, 3):
                    check_row, check_col = row + r_offset, col + c_offset
                    if 0 <= check_row < grid_h and 0 <= check_col < grid_w:
                        s = grid[check_row][check_col]
                        if s:
                            dist = math.hypot(s[0] - pt[0], s[1] - pt[1])
                            if dist < min_dist:
                                return False
            return True

        # Initial point
        start_pt = (random.uniform(0, self.size), random.uniform(0, self.size))
        while not is_valid(start_pt): # Should find one eventually
             start_pt = (random.uniform(0, self.size), random.uniform(0, self.size))

        process_list.append(start_pt)
        samples.append(start_pt)
        col, row = get_grid_coords(start_pt)
        grid[row][col] = start_pt
        
        head = 0
        while head < len(process_list):
            curr_pt = process_list[head]
            head += 1
            
            for _ in range(max_attempts):
                a = random.uniform(0, 2 * math.pi)
                r = random.uniform(min_dist, 2 * min_dist)
                
                new_pt_x = curr_pt[0] + r * math.cos(a)
                new_pt_y = curr_pt[1] + r * math.sin(a)
                new_pt = (new_pt_x, new_pt_y)
                
                if is_valid(new_pt):
                    process_list.append(new_pt)
                    samples.append(new_pt)
                    n_col, n_row = get_grid_coords(new_pt)
                    grid[n_row][n_col] = new_pt
        
        return samples
        
    def create_clutter_layer(self, base, min_dist=70):
        points = self.poisson_disk_sampling(min_dist=min_dist)
        for center in points:
            nodes, concave_prob = random.choice(self.clutter_config['node_variety'])
            size = self.sample_clutter_size()
            if np.random.rand() < concave_prob:
                poly = self.generate_concave_poly(center, nodes, size)
            else:
                poly = self.generate_convex_poly(center, nodes, size)
            if poly is not None and len(poly) > 0:
                cv2.fillPoly(base, [poly.astype(np.int32)], 0)
        return base

    def create_obstacle_layer(self, base, min_dist=50):
        points = self.poisson_disk_sampling(min_dist=min_dist)
        for center in points:
            nodes, concave_prob = random.choice(self.obstacle_config['node_variety'])
            size = self.sample_size()
            if np.random.rand() < concave_prob:
                poly = self.generate_concave_poly(center, nodes, size)
            else:
                poly = self.generate_convex_poly(center, nodes, size)
            if poly is not None and len(poly) > 0:
                cv2.fillPoly(base, [poly.astype(np.int32)], 0)
        return base

    def sample_clutter_size(self):
        min_s, max_s = self.clutter_config['size_range']
        if min_s >= max_s : return min_s # Avoid issues if range is too small
        x = np.arange(min_s, max_s +1)
        pdf = self.clutter_config['density_curve'](x)
        pdf_sum = pdf.sum()
        if pdf_sum == 0: return random.randint(min_s, max_s) # Fallback
        pdf /= pdf_sum
        return np.random.choice(x, p=pdf)

    def sample_size(self):
        min_s, max_s = self.obstacle_config['size_range']
        if min_s >= max_s: return min_s # Avoid issues
        x = np.arange(min_s, max_s + 1)
        pdf = self.obstacle_config['density_curve'](x)
        pdf_sum = pdf.sum()
        if pdf_sum == 0: return random.randint(min_s, max_s) # Fallback
        pdf /= pdf_sum
        return np.random.choice(x, p=pdf)

    def generate_concave_poly(self, center, nodes, size, 
                         max_concave_ratio=0.6, 
                         max_depth_ratio=0.7):
        if not (3 <= nodes <= 12): nodes = random.randint(3,12) # Ensure valid
        
        base_angles = np.sort(np.random.uniform(0, 2*np.pi, nodes))
        
        radii = np.random.weibull(1.5, nodes) * size * 0.6 + size*0.4
        radii = np.clip(radii, size*0.3, size*1.5) # Allow some extension
        
        concave_num = min(max(1, np.random.randint(1, int(nodes * max_concave_ratio) +1)), nodes // 2) # At least 1
        
        # Ensure concave_num is reasonable
        if nodes < 4 and concave_num > 0: concave_num = 1 # For triangles, at most 1 concave makes sense this way
        elif nodes < 4 : concave_num = 0


        concave_indices = self._select_concave_positions(nodes, concave_num)
        
        for i in concave_indices:
            depth_factor = np.random.uniform(0.2, max_depth_ratio)
            angle_shift_range = np.pi/(nodes*2)
            
            prev_idx = (i-1+nodes) % nodes
            next_idx = (i+1) % nodes
            
            anchor_angle = (base_angles[prev_idx] + base_angles[next_idx])/2
            if base_angles[next_idx] < base_angles[prev_idx]: # Handles wrap around 2pi
                anchor_angle = (base_angles[prev_idx] + base_angles[next_idx] + 2*np.pi)/2 % (2*np.pi)

            base_angles[i] = anchor_angle + np.random.uniform(-angle_shift_range, angle_shift_range)
            radii[i] *= depth_factor
            
        sorted_indices = np.argsort(base_angles)
        poly = self._create_polygon(center, base_angles[sorted_indices], radii[sorted_indices])
        
        if Polygon and make_valid: # Check if shapely is available
             return self._validate_concave_shape(poly)
        return poly.astype(np.int32)


    def _select_concave_positions(self, nodes, concave_num, min_gap=1): # min_gap=1 allows adjacent
        if concave_num == 0: return []
        if nodes == 0: return [] # prevent error if nodes is 0

        available_indices = list(range(nodes))
        random.shuffle(available_indices)
        
        selected_indices = []
        for idx in available_indices:
            is_valid = True
            for sel_idx in selected_indices:
                diff = abs(idx - sel_idx)
                # Check distance considering wrap-around for circular arrangement
                if min(diff, nodes - diff) < min_gap:
                    is_valid = False
                    break
            if is_valid:
                selected_indices.append(idx)
            if len(selected_indices) == concave_num:
                break
        return selected_indices

    def _validate_concave_shape(self, poly):
        if not Polygon or not make_valid: return poly.astype(np.int32) # Shapely not available

        if len(poly) < 3: return poly.astype(np.int32) # Not a polygon
        shapely_poly = Polygon(poly)
        if not shapely_poly.is_valid:
            repaired = make_valid(shapely_poly)
            if repaired.geom_type == 'Polygon':
                return np.array(repaired.exterior.coords[:-1], dtype=np.int32)
            elif repaired.geom_type == 'MultiPolygon': # Take largest
                largest_poly = max(repaired.geoms, key=lambda p: p.area)
                return np.array(largest_poly.exterior.coords[:-1], dtype=np.int32)
        return poly.astype(np.int32)

    def generate_convex_poly(self, center, nodes, base_radius, 
                        angle_perturb=0.3, radius_variance=0.4):
        if nodes < 3: nodes = 3 # Min 3 nodes for a polygon
        
        base_angles = np.linspace(0, 2*np.pi, nodes, endpoint=False)
        perturbed_angles = []
        current_angle_offset = 0
        
        # Generate angles with perturbation, ensuring they remain sorted relative to base_angles
        for i in range(nodes):
            # Max perturbation for this segment
            max_angle_pert = (2 * np.pi / nodes) * angle_perturb
            # Generate a perturbation within [-max_angle_pert, +max_angle_pert]
            angle_noise = np.random.uniform(-max_angle_pert, max_angle_pert)
            perturbed_angles.append(base_angles[i] + angle_noise)

        perturbed_angles = np.sort(np.array(perturbed_angles) % (2*np.pi)) # Sort and wrap

        radii = base_radius * np.random.uniform(1 - radius_variance, 
                                            1 + radius_variance, 
                                            size=nodes)
        radii = np.maximum(radii, base_radius * 0.2) # Ensure radius is not too small
        
        points = []
        for angle, r in zip(perturbed_angles, radii):
            x = center[0] + r * np.cos(angle)
            y = center[1] + r * np.sin(angle)
            points.append((x, y))
        
        poly = np.array(points, dtype=np.int32)
        
        try:
            hull = cv2.convexHull(poly)
            # Check if the number of points in the hull is close to the original number of nodes.
            # This is a simple check; for very perturbed shapes, it might simplify more.
            if len(hull) < nodes * 0.8 and nodes > 3 : # If too many points were lost
                 # Fallback: generate points directly on a circle if convexHull drastically simplified it
                return self.generate_convex_poly(center, nodes, base_radius, angle_perturb/2, radius_variance/2) # Try with less variance
            poly = hull.reshape(-1, 2) # Reshape hull output
        except Exception: # cv2.convexHull can fail on degenerate cases
            # Fallback: if convexHull fails, just return the points if they form a simple polygon
            pass # Return poly as is
            
        return poly.astype(np.int32)


    def _create_polygon(self, center, angles, radii):
        points = []
        for a, r in zip(angles, radii):
            x = center[0] + r * np.cos(a)
            y = center[1] + r * np.sin(a)
            points.append([x, y])
        return np.array(points) # Keep as float for potential shapely processing

    def morphological_detail(self, base):
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        kernel_erode_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)) # Smaller erosion

        temp = cv2.dilate(base, kernel_dilate, iterations=1)
        
        # Lighter random erosion
        for _ in range(2): # Reduced iterations
            rand_kernel_array = (np.random.rand(3,3) > 0.8).astype(np.uint8) # Sparser kernel
            if np.sum(rand_kernel_array) > 0: # Ensure kernel is not all zeros
                 temp = cv2.erode(temp, rand_kernel_array, iterations=1)
            else: # Fallback kernel if random one is empty
                 temp = cv2.erode(temp, kernel_erode_small, iterations=1)
        
        return cv2.dilate(temp, kernel_dilate, iterations=1)


class ConnectivityOptimizer:
    def __init__(self, size, channel_min_factor=0.03): # e.g. 30 for 1024px map
        self.size = size
        self.channel_min = max(5, int(size * channel_min_factor)) 
    
    def optimize(self, img):
        processed_img = self._preprocess(img.copy()) # Work on a copy
        
        graph, labels = self._build_region_graph(processed_img)
        if graph is None or graph.number_of_nodes() <=1: # No regions or only one region
            if isinstance(processed_img, np.ndarray) and processed_img.dtype == bool:
                 return processed_img.astype(np.uint8) * 255
            return processed_img # Already fine or nothing to do

        repaired_img = self._global_repair(processed_img, graph, labels)
        
        # Ensure output is uint8
        if repaired_img.dtype == bool:
            return repaired_img.astype(np.uint8) * 255
        return repaired_img


    def _preprocess(self, img):
        # Ensure img is uint8
        if img.dtype == bool:
            img_uint8 = img.astype(np.uint8) * 255
        elif img.dtype != np.uint8:
            img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_uint8 = img
        
        # Create boundary, making it non-traversable (0) for connectedComponents
        # Thickness of boundary should be less than channel_min to not be erased by carve_path
        boundary_thickness = max(1, self.channel_min // 4) 
        cv2.rectangle(img_uint8, (0,0), (self.size-1, self.size-1), 0, boundary_thickness)

        # Binarize: free space is 255 (white), obstacles are 0 (black)
        _, binarized = cv2.threshold(img_uint8, 127, 255, cv2.THRESH_BINARY)
        
        # Invert for connectedComponents: components are obstacles (0), background is traversable (255)
        # No, connectedComponents should find free space. So, binarized is correct.
        # Free space = 255, Obstacles = 0
        return binarized # Return as uint8

    def _build_region_graph(self, img_uint8): # Expects uint8 image
        # Find connected components of FREE SPACE (255)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_uint8, connectivity=8)
        
        graph = nx.Graph()
        if num_labels <= 1: # Only background or no components
            return graph, labels

        # Label 0 is the background (largest component if map is mostly obstacles, or if inverted)
        # Assuming free space is white (255), components are free areas.
        # We usually want to connect these free areas.
        
        for i in range(1, num_labels): # Skip label 0 (often largest background if inverted)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.channel_min * self.channel_min / 4: # Filter out very small components
                continue
            # Get contour by creating a mask for the current label
            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours: # Ensure contour exists
                graph.add_node(i, contour=contours[0], area=area, centroid=centroids[i])
        
        return graph, labels


    def _global_repair(self, img_uint8, graph, labels_map): # Expects uint8 image
        if graph.number_of_nodes() == 0:
             return img_uint8 # No components to connect

        components = list(nx.connected_components(graph)) # This graph is node-per-region, so initially all are disconnected
                                                        # We need to build adjacencies or connect them
        
        # Identify the main component (largest free space area)
        if not components: # Should not happen if graph has nodes
            return img_uint8

        # Iterate while there's more than one "super-component" of free space that we want to connect
        # This loop is for connecting disconnected free space regions
        # Let's find the largest component by area
        if graph.number_of_nodes() <= 1:
            return img_uint8 # Already connected or single component
            
        all_nodes = list(graph.nodes(data=True))
        if not all_nodes: return img_uint8

        # Sort nodes by area to find the largest component (main free space)
        all_nodes.sort(key=lambda x: x[1]['area'], reverse=True)
        main_component_node_id = all_nodes[0][0]
        
        # Connect other components to the main one
        for i in range(1, len(all_nodes)):
            target_node_id = all_nodes[i][0]
            
            # Find path between centroids (simple approach)
            p1 = graph.nodes[main_component_node_id]['centroid']
            p2 = graph.nodes[target_node_id]['centroid']
            
            # Carve path on the uint8 image
            img_uint8 = self._carve_path(img_uint8, (p1, p2))

        return img_uint8


    def _find_optimal_path(self, graph, main_comp_nodes, target_comp_nodes):
        # This function assumes main_comp_nodes and target_comp_nodes are lists of node IDs
        main_points = [graph.nodes[n]['centroid'] for n in main_comp_nodes if n in graph]
        target_points = [graph.nodes[n]['centroid'] for n in target_comp_nodes if n in graph]

        if not main_points or not target_points:
            return None # Cannot find path if one set is empty
            
        min_dist = np.inf
        optimal_pair = None
        for p1 in main_points:
            for p2 in target_points:
                dist = np.linalg.norm(np.array(p1)-np.array(p2))
                if dist < min_dist:
                    min_dist = dist
                    optimal_pair = (p1, p2)
        return optimal_pair

    def _carve_path(self, img_uint8, path_points): # Expects uint8 image
        p1, p2 = path_points
        # Draw a white line (255) for the path
        cv2.line(img_uint8, 
                 (int(p1[0]), int(p1[1])), 
                 (int(p2[0]), int(p2[1])), 
                 255,  # Carve with white (free space)
                 self.channel_min) # Thickness of the path
        
        # Optional: A bit of morphology to smooth the path, but might be too much
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.channel_min//2, self.channel_min//2))
        # img_uint8 = cv2.dilate(img_uint8, kernel) 
        # img_uint8 = cv2.erode(img_uint8, kernel)
        return img_uint8

    def _get_contour_center(self, contour):
        M = cv2.moments(contour)
        if M['m00'] == 0: # Avoid division by zero
            # Fallback to bounding rect center if moment is zero
            x,y,w,h = cv2.boundingRect(contour)
            return (x+w/2, y+h/2)
        return (M['m10']/M['m00'], M['m01']/M['m00'])
    

# PathGenerator class (mostly unchanged, but removed torch dependencies for broader compatibility here)
class PathGenerator:
    def __init__(self, bitmap, agent_radius_px): # agent_radius in pixels
        self.bitmap = bitmap # Expected to be 0 for obstacle, 255 for free
        self.agent_radius_px = agent_radius_px
       
    def _is_valid_point(self, coord, bitmap):
        y, x = int(coord[0]), int(coord[1])
        if 0 <= y < bitmap.shape[0] and 0 <= x < bitmap.shape[1]:
            return bitmap[y, x] == 255 
        return False

    def _is_safe_point(self, coord, bitmap, safety_radius_px):
        y, x = int(coord[0]), int(coord[1])
        radius_int = int(np.ceil(safety_radius_px))
        
        for dy in range(-radius_int, radius_int + 1):
            for dx in range(-radius_int, radius_int + 1):
                if dx**2 + dy**2 > safety_radius_px**2:
                    continue
                check_y, check_x = y + dy, x + dx
                if 0 <= check_y < bitmap.shape[0] and 0 <= check_x < bitmap.shape[1]:
                    if bitmap[check_y, check_x] == 0:  # Obstacle
                        return False
        return True

    def _find_path_a_star(self, start, goal, bitmap, safety_radius_px=0):
        import heapq
        
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        closed_set = set()
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if np.linalg.norm(np.array(current) - np.array(goal)) < 2: # Close enough
                path = []
                temp = current
                while temp in came_from:
                    path.append(temp)
                    temp = came_from[temp]
                path.append(start)
                return path[::-1]
            
            closed_set.add(current)
            
            for dy, dx in neighbors:
                neighbor = (current[0] + dy, current[1] + dx)
                
                if neighbor in closed_set:
                    continue
                if not self._is_valid_point(neighbor, bitmap) or \
                   not self._is_safe_point(neighbor, bitmap, safety_radius_px):
                    continue
                    
                movement_cost = np.sqrt(dx*dx + dy*dy)
                tentative_g_score = g_score[current] + movement_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def get_random_path(self, max_trials=100): # Simplified path generation
        h, w = self.bitmap.shape
        
        start_coord, target_coord = None, None

        for _ in range(max_trials): # Find valid start
            sy, sx = random.randint(0, h-1), random.randint(0, w-1)
            if self._is_valid_point((sy,sx), self.bitmap) and self._is_safe_point((sy,sx), self.bitmap, self.agent_radius_px):
                start_coord = (sy, sx)
                break
        if not start_coord: return []

        for _ in range(max_trials): # Find valid target
            ty, tx = random.randint(0, h-1), random.randint(0, w-1)
            if self._is_valid_point((ty,tx), self.bitmap) and self._is_safe_point((ty,tx), self.bitmap, self.agent_radius_px):
                dist_sq = (sy-ty)**2 + (sx-tx)**2
                if dist_sq > (max(h,w)/4)**2: # Ensure target is reasonably far
                    target_coord = (ty, tx)
                    break
        if not target_coord: return []
            
        print(f"Attempting path from {start_coord} to {target_coord} with agent_radius_px {self.agent_radius_px}")
        path = self._find_path_a_star(start_coord, target_coord, self.bitmap, safety_radius_px=self.agent_radius_px)
        
        if path:
            # Path points are (y,x) pixel coordinates
            return [(p[1], p[0]) for p in path] # Convert to (x,y) for consistency if needed elsewhere
        return []


def main():
    parser = argparse.ArgumentParser(description='生成高连通性隧道导航地图')
    parser.add_argument('--num', type=int, default=200, help='生成地图数量，默认5')
    parser.add_argument('--size', type=int, default=256, choices=[256, 512, 768, 1024, 2048], help='地图尺寸（像素），默认256')
    parser.add_argument('--output', type=str, default='../train_tunnel_maps_2', help='输出目录路径，默认./tunnel_maps')
    parser.add_argument('--version', action='version', version='tunnel-v1.0.0')
    parser.add_argument('--type', type=str, default='tunnel', choices=['obstacle_and_clutter', 'tunnel', 'obstacle', 'clutter'], help="Map type")
    parser.add_argument('--level', type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help="Difficulty level")
    parser.add_argument('--path', type=int, default=0, help='0-no path generated, 1-generate paths')

    # Tunnel specific parameters
    parser.add_argument('--tunnel_start_x_min', type=float, default=0.57, help="Min X start for tunnel (fraction of size), must be > 0.5")
    parser.add_argument('--tunnel_start_x_max', type=float, default=0.63, help="Max X start for tunnel (fraction of size)")
    parser.add_argument('--tunnel_height_min', type=float, default=0.07, help="Min tunnel height (fraction of size)")
    parser.add_argument('--tunnel_height_max', type=float, default=0.12, help="Max tunnel height (fraction of size)")
    
    args = parser.parse_args()

    if args.tunnel_start_x_min <= 0.5:
        print("Warning: tunnel_start_x_min should be > 0.5 to ensure tunnel is in the right half. Adjusting to 0.51.")
        args.tunnel_start_x_min = 0.51
    if args.tunnel_start_x_max <= args.tunnel_start_x_min:
        args.tunnel_start_x_max = args.tunnel_start_x_min + 0.1
        print(f"Warning: tunnel_start_x_max adjusted to {args.tunnel_start_x_max}")
    if args.tunnel_height_max <= args.tunnel_height_min:
        args.tunnel_height_max = args.tunnel_height_min + 0.05
        print(f"Warning: tunnel_height_max adjusted to {args.tunnel_height_max}")


    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"初始化{args.size}x{args.size}隧道地图生成器...")
    generator = TunnelMapGenerator(
        size=args.size,
        tunnel_start_x_range=(args.tunnel_start_x_min, args.tunnel_start_x_max),
        tunnel_height_range=(args.tunnel_height_min, args.tunnel_height_max)
    )
    
    success_count = 0
    start_time = time.time()
    
    while success_count < args.num:
        map_id = success_count + 1 # Attempt ID
        print(f"\n生成地图 #{map_id}/{args.num}")
        try:
            map_data = generator.generate_map(level=args.level, type=args.type)
            
            # Pass generator instance for validate_map to access tunnel params
            if validate_map(map_data, map_type=args.type, generator_instance=generator):
                filename = output_dir / f"map_{args.type}_{map_id}_{args.size}px.png"
                cv2.imwrite(str(filename), map_data)
                print(f"已保存：{filename.name}")
                
                if args.path == 1:
                    path_num_to_generate = 5
                    paths_found = []
                    # agent_radius_world = 0.5 # Example world units
                    # scale = 0.1 # Example: 0.1 world units per pixel
                    # agent_radius_pixels = int(agent_radius_world / scale)
                    agent_radius_pixels = int(args.size * 0.02) # e.g. 2% of map size for agent radius in pixels
                    agent_radius_pixels = max(3, agent_radius_pixels) # Min 3 pixels

                    print(f"Generating paths with agent_radius_pixels: {agent_radius_pixels}")
                    path_gen = PathGenerator(map_data, agent_radius_pixels)
                    for n in range(path_num_to_generate):
                        print(f"  Generating path {n+1}/{path_num_to_generate} for {filename.stem}")
                        single_path = path_gen.get_random_path()
                        if single_path:
                            paths_found.append(single_path)
                        else:
                            print(f"    Could not find path {n+1}")
                    
                    if paths_found:
                        import pickle
                        path_filename = output_dir / f"{filename.stem}_paths.pkl"
                        with open(path_filename, 'wb') as f:
                            pickle.dump(paths_found, f)
                        print(f"  Saved {len(paths_found)} paths to {path_filename.name}")
                
                success_count += 1 # Only increment if valid and saved
            else:
                print("地图质量检查失败，重新生成...")
                
        except Exception as e:
            print(f"生成过程中发生错误：{str(e)}")
            import traceback
            traceback.print_exc()
            # Optionally, decide if you want to retry or break on error
            # For now, it will just print error and the loop will continue to try next map
            # if too many errors, consider adding a counter to break
            continue

    total_time = time.time() - start_time
    if args.num > 0:
        print(f"\n生成完成！成功 {success_count}/{args.num} 张. 平均耗时：{total_time/args.num:.2f}s/张 (基于尝试次数)")
    else:
        print("\n没有请求生成地图.")
    print(f"输出目录：{output_dir.resolve()}")


def validate_map(img, map_type="obstacle_and_clutter", generator_instance=None):
    if img is None or img.size == 0:
        print("验证失败: 图像为空.")
        return False
        
    if img.dtype == bool: # Ensure uint8 for processing
        binary_map = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        binary_map = np.clip(img, 0, 255).astype(np.uint8)
    else:
        binary_map = img
    
    # Ensure it's truly binary (0 or 255) for checks
    _, process_map = cv2.threshold(binary_map, 127, 255, cv2.THRESH_BINARY)

    # Obstacle ratio (0 is obstacle, 255 is free)
    obstacle_ratio = np.mean(process_map == 0)
    
    min_ratio, max_ratio = 0.2, 0.7
    if map_type == "tunnel":
        min_ratio, max_ratio = 0.0, 0.95 # Tunnels have more fixed obstacles on one side

    if not (min_ratio < obstacle_ratio < max_ratio):
        print(f"障碍物比例异常：{obstacle_ratio:.2f} (期望范围: {min_ratio}-{max_ratio})")
        return False
    
    height, width = process_map.shape
    
    if map_type == "tunnel":
        # Use actual tunnel parameters if available from generator_instance
        tunnel_start_x = getattr(generator_instance, 'current_tunnel_start_x', int(width * 0.6))
        tunnel_h = getattr(generator_instance, 'current_tunnel_height', int(height * 0.15))
        
        tunnel_y_mid = height // 2
        tunnel_entrance_x = min(width -1, tunnel_start_x + 5) # A bit inside the tunnel
        tunnel_exit_x = max(0, width - 10) # Near the right edge

        # Check if key tunnel points are free
        if process_map[tunnel_y_mid, tunnel_entrance_x] == 0: # Obstacle
            print(f"隧道入口 ({tunnel_entrance_x},{tunnel_y_mid}) 被阻塞.")
            return False
        if process_map[tunnel_y_mid, tunnel_exit_x] == 0: # Obstacle
            print(f"隧道出口 ({tunnel_exit_x},{tunnel_y_mid}) 被阻塞.")
            return False

        # Check path from a point on left to a point in the tunnel exit
        start_pt_left = (height // 2, max(0, tunnel_start_x // 2)) # Middle of left area
        if process_map[start_pt_left[0], start_pt_left[1]] == 0: # If start is blocked, try another
            start_pt_left = (height // 2, max(0, int(width*0.1))) # Further left
            if process_map[start_pt_left[0], start_pt_left[1]] == 0:
                 print(f"左侧起始点 ({start_pt_left[1]},{start_pt_left[0]}) 被阻塞.")
                 return False # Can't even start path check

        end_pt_tunnel = (tunnel_y_mid, tunnel_exit_x)
        if not has_path(process_map == 255, start_pt_left, end_pt_tunnel): # has_path expects True for free
            print(f"无法从左侧 ({start_pt_left[1]},{start_pt_left[0]}) 到达隧道出口 ({end_pt_tunnel[1]},{end_pt_tunnel[0]}).")
            return False
    else: # Standard map validation (original logic)
        center_y, center_x = height // 2, width // 2
        central_clear_size = getattr(generator_instance, 'central_clear_size', int(height*0.06))

        if process_map[center_y, center_x] == 0:
            print("中央区域被障碍物覆盖 (标准地图).")
            return False
        
        random_positions = []
        attempts = 0
        num_pts_to_check = 3
        while len(random_positions) < num_pts_to_check and attempts < 100:
            end_y, end_x = random.randint(0, height - 1), random.randint(0, width - 1)
            outside_central = (abs(end_y - center_y) > central_clear_size//2 or \
                               abs(end_x - center_x) > central_clear_size//2)
            if outside_central and process_map[end_y, end_x] == 255: # Free space
                random_positions.append((end_y, end_x))
            attempts += 1
        
        if len(random_positions) < num_pts_to_check: # Could be very dense or very sparse
            print(f"找不到足够的可访问随机位置 ({len(random_positions)}/{num_pts_to_check}).")
            # This might not be a hard fail for some map types, but indicates potential issues.
            # For now, let's not fail here if at least one point was found for less dense maps.
            if not random_positions: return False


        for i, (end_y, end_x) in enumerate(random_positions):
            if not has_path(process_map == 255, (center_y, center_x), (end_y, end_x)):
                print(f"中央区域无法到达随机位置 {i+1}: ({end_x}, {end_y}) (标准地图).")
                return False
    
    print("地图质量检查通过.")
    return True

def has_path(grid_bool, start_yx, end_yx): # Expects boolean grid (True is traversable)
    height, width = grid_bool.shape
    q = [(start_yx)]
    visited = np.zeros_like(grid_bool, dtype=bool)
    
    if not grid_bool[start_yx[0], start_yx[1]]: return False # Start is obstacle
    visited[start_yx[0], start_yx[1]] = True
    
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
    
    head = 0
    while head < len(q):
        curr_y, curr_x = q[head]
        head+=1
        
        if (curr_y, curr_x) == end_yx:
            return True
        
        for dy, dx in moves:
            ny, nx = curr_y + dy, curr_x + dx
            if 0 <= ny < height and 0 <= nx < width and \
               grid_bool[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    return False


if __name__ == "__main__":
    main()