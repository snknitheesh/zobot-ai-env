import yaml
import os
import glob
import cv2
from tqdm import tqdm
import base64
import numpy as np
import Definitions
import fnmatch
import matplotlib.pyplot as plt
from data_loader import DataLoader

class SemanticProcessor:
    def __init__(self, yaml_file='data.yaml'):
        self.data_loader = DataLoader(yaml_file)

    def create_road_and_anomaly_overlay(self, semantic_img, rgb_img):
        """
        Create road and anomaly overlay visualization from semantic and RGB images using colors from Definitions.LABELS.
        """
        road_color = np.array([label.color for label in Definitions.LABELS if label.name == "road"][0])
        road_color_bgr = (road_color[2], road_color[1], road_color[0])
        
        anomaly_labels = [label for label in Definitions.LABELS if 
                         label.name in ["anomaly", "home", "animal", "nature", 
                                      "special", "falling", "airplane"]]
        anomaly_ids = [label.id for label in anomaly_labels]
        
        road_mask = np.zeros(semantic_img.shape[:2], dtype=np.uint8)
        anomaly_mask = np.zeros(semantic_img.shape[:2], dtype=np.uint8)
        
        road_mask[semantic_img[:,:,0] == 1] = 255
        
        for anomaly_id in anomaly_ids:
            anomaly_mask[semantic_img[:,:,0] == anomaly_id] = 255
        
        road_colored_mask = np.zeros_like(rgb_img)
        anomaly_colored_mask = np.zeros_like(rgb_img)
        road_colored_mask[road_mask == 255] = road_color_bgr
        
        anomaly_color = np.array(anomaly_labels[0].color)
        anomaly_color_bgr = (anomaly_color[2], anomaly_color[1], anomaly_color[0])
        anomaly_colored_mask[anomaly_mask == 255] = anomaly_color_bgr
        
        road_overlay = cv2.addWeighted(rgb_img, 1.0, road_colored_mask, 0.5, 0)
        anomaly_overlay = cv2.addWeighted(rgb_img, 1.0, anomaly_colored_mask, 0.5, 0)
        
        combined_mask = np.zeros_like(rgb_img)
        combined_mask[road_mask == 255] = road_color_bgr
        combined_mask[anomaly_mask == 255] = anomaly_color_bgr
        combined_overlay = cv2.addWeighted(rgb_img, 1.0, combined_mask, 0.5, 0)
        
        return road_mask, anomaly_mask, road_overlay, anomaly_overlay, combined_mask, combined_overlay

    def visualize_segmentation(self, original_img, converted_img, rgb_img, window_name="Visualization", current_idx=None, total=None):
        """
        Visualize the images in two rows:
        Row 1: [RGB | Labeled Semantic | Labeled Semantic (Road Only + Actors on Road)]
        Row 2: [Road Mask | Anomaly Mask | Combined Overlay]
        """
        if converted_img.dtype == np.float32 or converted_img.dtype == np.float64:
            converted_img = (converted_img * 255).astype(np.uint8)
        
        road_mask, anomaly_mask, road_overlay, anomaly_overlay, combined_mask, combined_overlay = self.create_road_and_anomaly_overlay(original_img, rgb_img)
        
        img_width = original_img.shape[1]
        img_height = original_img.shape[0]
        canvas_width = img_width * 3
        canvas_height = img_height * 2
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Row 1
        canvas[:img_height, :img_width] = rgb_img  # RGB
        canvas[:img_height, img_width:img_width*2] = converted_img  # Labeled Semantic

        # --- NEW: Labeled Semantic (Road Only + Actors on Road) ---
        # Find the road class id
        road_label = [label for label in Definitions.LABELS if label.name == "road"][0]
        road_id = road_label.id

        # Create a binary mask for the road
        road_region = (original_img[:, :, 0] == road_id).astype(np.uint8)

        # Dilate the road mask to include actors touching the road
        kernel = np.ones((15, 15), np.uint8)  # You can adjust the kernel size
        road_region_dilated = cv2.dilate(road_region, kernel, iterations=1)

        # Create output image: keep all labeled semantic pixels where the dilated road mask is 1
        labeled_semantic_road_and_actors = np.zeros_like(converted_img)
        mask = road_region_dilated.astype(bool)
        labeled_semantic_road_and_actors[mask] = converted_img[mask]
        canvas[:img_height, img_width*2:] = labeled_semantic_road_and_actors  # Labeled Semantic (Road+Actors)

        # Row 2
        road_mask_3ch = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
        anomaly_mask_3ch = cv2.cvtColor(anomaly_mask, cv2.COLOR_GRAY2BGR)
        canvas[img_height:, :img_width] = road_mask_3ch  # Road Mask
        canvas[img_height:, img_width:img_width*2] = anomaly_mask_3ch  # Anomaly Mask
        canvas[img_height:, img_width*2:] = combined_overlay  # Combined Overlay

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        row1_labels = ["RGB", "Labeled Semantic", "Labeled Semantic (Road+Actors)"]
        for i, label in enumerate(row1_labels):
            cv2.putText(canvas, label, 
                       (i * img_width + 10, 30),
                       font, font_scale, (255, 255, 255), thickness)

        row2_labels = ["Road Mask", "Anomaly Mask", "Combined Overlay"]
        for i, label in enumerate(row2_labels):
            cv2.putText(canvas, label, 
                       (i * img_width + 10, img_height + 30),
                       font, font_scale, (255, 255, 255), thickness)

        if current_idx is not None and total is not None:
            cv2.putText(canvas, f"Image {current_idx + 1}/{total} (ESC to exit)", 
                       (10, canvas_height - 10), font, font_scale, (255, 255, 255), thickness)

        cv2.imshow(window_name, canvas)
        return cv2.waitKey(0)

    def process_and_visualize(self):
        """
        Main method to process and visualize the semantic segmentation
        """
        paths = self.data_loader.get_data_paths('scenario')
        
        semantic_cam_path = self.data_loader.find_directory("SEMANTIC-CAM*", paths)
        rgb_cam_path = self.data_loader.find_directory("RGB-CAM*", paths)
        
        if not semantic_cam_path or not rgb_cam_path:
            raise ValueError("Could not find required camera directories")
        
        # Load and process images using the DataLoader
        sem_sorted, rgb_sorted, _, _ = self.data_loader.load_images(
            semantic_cam_path=semantic_cam_path,
            rgb_cam_path=rgb_cam_path
        )
        
        if not sem_sorted or not rgb_sorted:
            raise ValueError("Failed to load semantic or RGB images")
        
        print("\nStarting visualization...")
        print("Use Left/Right arrow keys to navigate through images")
        print("Press ESC to exit")
        
        current_idx = 0
        while True:
            _, _, org_img, converted_img, img_road_actors = sem_sorted[current_idx]
            _, rgb_img = rgb_sorted[current_idx]
            
            key = self.visualize_segmentation(org_img, converted_img, rgb_img,
                                           current_idx=current_idx, 
                                           total=len(sem_sorted))
            
            if key == 27: 
                break
            elif key in [81, 2424832]:  
                current_idx = (current_idx - 1) % len(sem_sorted)
            elif key in [83, 2555904]: 
                current_idx = (current_idx + 1) % len(sem_sorted)
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    processor = SemanticProcessor()
    processor.process_and_visualize()





