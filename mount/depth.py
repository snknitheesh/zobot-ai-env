import cv2
import numpy as np
from data_loader import DataLoader
import base64


data_loader = DataLoader()

paths = data_loader.get_data_paths('scenario')
depth_cam_path = data_loader.find_directory("DEPTH_CAM*", paths)

if depth_cam_path is None:
    print("Error: Depth camera directory not found")
    exit()

print(f"Found depth camera path: {depth_cam_path}")

_, _, depth_data, _ = data_loader.load_images(
    semantic_cam_path=None,
    rgb_cam_path=None,
    depth_cam_path=depth_cam_path,
    instance_cam_path=None
)

if depth_data is None or len(depth_data) == 0:
    print("No depth images found in the specified path")
    exit()

print(f"Loaded {len(depth_data)} depth images")

depth_frames = list(depth_data)
current_frame = 0
total_frames = len(depth_frames)

window_name = 'Depth Viewer (Left/Right arrows to navigate, Q to quit)'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

def show_current_frame():
    frame_id, img_encoded, org_img = depth_frames[current_frame]
    img_bytes = base64.b64decode(img_encoded)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    text = f"Frame: {current_frame + 1}/{total_frames}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow(window_name, img)

show_current_frame()

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'): 
        break
    elif key == 81 or key == 2 or key == ord('a'):  
        current_frame = (current_frame - 1) % total_frames
        show_current_frame()
    elif key == 83 or key == 3 or key == ord('d'):  
        current_frame = (current_frame + 1) % total_frames
        show_current_frame()

cv2.destroyAllWindows()


