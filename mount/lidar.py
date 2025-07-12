import numpy as np
import open3d as o3d
from data_loader import DataLoader, Definitions
import sys
from tqdm import tqdm

POINT_SIZE = 2
VOXEL_RANGE = [[-50, 50], [-50, 50], [-4, 10]]
VOXEL_EXPAND_FACTOR = 2
PRUNE = 1
ANOMALY_IDS = [29, 30, 31, 32, 33, 34, 100]
GREY_COLOR = (0.5, 0.5, 0.5)

vis = None
view_control = None
coord_frame = None
current_frame = 0
point_cloud_data = None
pcd = None
is_semantic = False
color_palette = None
view_mode = "semantic"  # "semantic", "anomaly", "road_only"

def initialize_visualizer():
    global vis, view_control, coord_frame, color_palette
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Anomaly Detection View", width=1600, height=900)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = POINT_SIZE
    
    color_palette = {label.id: (label.color[0]/255.0, label.color[1]/255.0, label.color[2]/255.0)
                    for label in Definitions.LABELS}
    
    view_control = vis.get_view_control()
    view_control.set_zoom(3.0)
    view_control.set_front([-0.5, -0.5, 1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 0, 1])
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    
    grid = o3d.geometry.TriangleMesh.create_box(width=200, height=200, depth=0.1)
    grid.compute_vertex_normals()
    grid.paint_uniform_color([0.1, 0.1, 0.1])
    grid.translate([-100, -100, -2])
    vis.add_geometry(grid)
    
    vis.register_key_callback(262, next_frame)  # Right arrow
    vis.register_key_callback(263, prev_frame)  # Left arrow
    vis.register_key_callback(32, next_frame)   # Space
    vis.register_key_callback(81, quit_vis)     # Q
    vis.register_key_callback(83, toggle_semantic_view)  # S - semantic view
    vis.register_key_callback(65, toggle_anomaly_view)   # A - anomaly view  
    vis.register_key_callback(82, toggle_road_view)      # R - road only view
    vis.register_key_callback(73, show_info)             # I - show info     

def load_data():
    """Load LiDAR data"""
    global point_cloud_data, is_semantic
    try:
        data_loader = DataLoader()
        paths = data_loader.get_data_paths('scenario')
        
        if not paths:
            raise ValueError("No scenario paths found in configuration")
        
        semantic_lidar_path = data_loader.find_directory("SEMANTIC-LIDAR*", paths)
        
        if not semantic_lidar_path:
            raise ValueError("No semantic LiDAR data directory found")
        
        _, point_cloud_data = data_loader.load_lidar_data(
            semantic_lidar_path=semantic_lidar_path,
            prune=PRUNE,
            voxel_range=VOXEL_RANGE,
            voxel_expand_factor=VOXEL_EXPAND_FACTOR
        )
        
        if point_cloud_data is not None:
            is_semantic = True
            print("Using semantic LiDAR data")
            print_semantic_info()
            visualize_frame()
        else:
            raise ValueError("No semantic LiDAR data loaded")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def get_semantic_colors(semantic_labels, highlight_anomalies=False):
    """Convert semantic labels to colors using CARLA color palette"""
    colors = np.zeros((len(semantic_labels), 3))
    
    for i, label in enumerate(semantic_labels):
        label_id = int(label)
        
        # Use proper color from palette if available
        if label_id in color_palette:
            colors[i] = color_palette[label_id]
        else:
            colors[i] = GREY_COLOR  # Unknown labels get grey
            
        # Highlight anomalies with brighter colors if requested
        if highlight_anomalies and label_id in ANOMALY_IDS:
            # Make anomaly colors brighter/more vivid
            colors[i] = np.clip(colors[i] * 1.5, 0.0, 1.0)
    
    return colors

def get_anomaly_colors(semantic_labels):
    """Convert semantic labels to colors, highlighting only anomalies (legacy function)"""
    colors = np.zeros((len(semantic_labels), 3))
    colors.fill(0.5)  
    
    for i, label in enumerate(semantic_labels):
        label_id = int(label)
        if label_id in ANOMALY_IDS:
            colors[i] = color_palette[label_id]
    return colors

def get_road_only_colors(semantic_labels):
    """Show only road surfaces (label ID 1) with proper color, everything else grey"""
    colors = np.zeros((len(semantic_labels), 3))
    colors.fill(0.2)  # Dark grey for non-road
    
    for i, label in enumerate(semantic_labels):
        label_id = int(label)
        if label_id == 1:  # Road
            colors[i] = color_palette[1] if 1 in color_palette else (0.5, 0.3, 0.5)  # Purple for road
    
    return colors

def visualize_frame():
    """Visualize the current frame"""
    global pcd
    if point_cloud_data is None or current_frame >= len(point_cloud_data):
        return
        
    points = point_cloud_data[current_frame]
    
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    if is_semantic:
        # Use semantic colors for all labels
        if view_mode == "semantic":
            colors = get_semantic_colors(points[:, 3], highlight_anomalies=False)
        elif view_mode == "anomaly":
            colors = get_anomaly_colors(points[:, 3])
        elif view_mode == "road_only":
            colors = get_road_only_colors(points[:, 3])
        else:
            # Default to semantic view
            colors = get_semantic_colors(points[:, 3], highlight_anomalies=False)
        
        new_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    if pcd is not None:
        vis.remove_geometry(pcd, False)
    
    pcd = new_pcd
    vis.add_geometry(pcd, False)
    
    vis.update_renderer()
    print(f"\rFrame: {current_frame + 1}/{len(point_cloud_data)} | Mode: {view_mode.upper()}", end="")

def next_frame(vis):
    """Show next frame"""
    global current_frame
    if point_cloud_data is not None and current_frame < len(point_cloud_data) - 1:
        current_frame += 1
        visualize_frame()
    return False

def prev_frame(vis):
    """Show previous frame"""
    global current_frame
    if point_cloud_data is not None and current_frame > 0:
        current_frame -= 1
        visualize_frame()
    return False

def quit_vis(vis):
    """Quit visualizer"""
    print("\nClosing visualizer...")
    vis.destroy_window()
    return True

def print_instructions():
    """Print usage instructions"""
    print("\nControls:")
    print("Right Arrow/Space: Next frame")
    print("Left Arrow: Previous frame")
    print("S: Semantic view (all labels with proper colors)")
    print("A: Anomaly view (only anomalies highlighted)")
    print("R: Road view (only road surfaces highlighted)")
    print("I: Show semantic label information")
    print("Q: Quit")
    print(f"\nCurrent mode: {view_mode.upper()}")

def toggle_semantic_view(vis):
    """Toggle to full semantic view"""
    global view_mode
    view_mode = "semantic"
    print("\nSemantic view: All labels with proper colors")
    visualize_frame()
    return False

def toggle_anomaly_view(vis):
    """Toggle to anomaly-only view"""
    global view_mode
    view_mode = "anomaly"
    print("\nAnomaly view: Only anomalies highlighted")
    visualize_frame()
    return False

def toggle_road_view(vis):
    """Toggle to road-only view"""
    global view_mode
    view_mode = "road_only"
    print("\nRoad view: Only road surfaces highlighted")
    visualize_frame()
    return False

def print_semantic_info():
    """Print information about semantic labels and their colors"""
    print("\nSemantic Label Information:")
    print("=" * 50)
    for label in Definitions.LABELS[:15]:  # Show first 15 labels
        color_rgb = [int(c) for c in label.color]
        print(f"ID {label.id:2d}: {label.name:15s} - Color: RGB{color_rgb}")
    
    print(f"\nAnomaly IDs: {ANOMALY_IDS}")
    print(f"Road ID: 1")
    print("=" * 50)

def show_info(vis):
    """Show semantic label information"""
    print_semantic_info()
    return False

def main():
    try:
        initialize_visualizer()
        load_data()
        print_instructions()
        vis.run()
    except Exception as e:
        print(f"Error running visualizer: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
