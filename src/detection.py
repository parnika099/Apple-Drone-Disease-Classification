import torch
import cv2
import numpy as np
import os
import time
import json
import warnings
from google.cloud import storage
from datetime import datetime
import io
import traceback
from pathlib import Path

# Suppress the FutureWarning about torch.cuda.amp.autocast deprecation
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")

# Set Google Cloud credentials - make robust to CWD
credentials_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src/google_credentials.json")
if os.path.exists(credentials_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
else:
    # Try alternate location or fallback
    if os.path.exists("src/google_credentials.json"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "src/google_credentials.json"

# Google Cloud Storage configuration
BUCKET_NAME = "apple-detection-images"
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize Google Cloud Storage client
try:
    storage_client = storage.Client()
    bucket = None
except Exception as e:
    print(f"Error initializing GCS client: {e}")
    storage_client = None

def get_bucket():
    global bucket
    if bucket is None and storage_client:
        try:
            bucket = storage_client.get_bucket(BUCKET_NAME)
            print(f"Using existing bucket: {BUCKET_NAME}")
        except Exception as e:
            print(f"Error accessing bucket: {str(e)}")
            print("Creating new bucket...")
            try:
                bucket = storage_client.create_bucket(BUCKET_NAME)
                print(f"Bucket {BUCKET_NAME} created successfully")
            except Exception as e:
                print(f"Failed to create bucket: {str(e)}")
    return bucket

# Load a specific YOLOv5 model for better performance
print("Loading YOLOv5 model...")
try:
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    # Set confidence threshold
    model.conf = 0.4  # Confidence threshold
    model.classes = [47]  # COCO class ID for apple is 47
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Ensure output directory exists
output_dir = "cropped_apples"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store apple count data and metadata
apple_count_data = {
    "frame_counts": [],
    "session_id": session_id,
    "timestamp": datetime.now().isoformat(),
    "cloud_image_urls": []
}

# Set threshold for bounding box overlap
IOU_THRESHOLD = 0.4

# Time window to reset tracking (in seconds)
TRACKING_RESET_TIME = 10  


# Function to upload image to GCS
def upload_to_gcs(image, image_name):
    """Upload an image to Google Cloud Storage"""
    if not get_bucket():
        return None
        
    try:
        # Convert OpenCV image to bytes
        is_success, buffer = cv2.imencode(".jpg", image)
        if not is_success:
            print(f"Failed to encode image: {image_name}")
            return None
        
        io_buf = io.BytesIO(buffer)
        
        # Upload to GCS
        blob_name = f"{session_id}/{image_name}"
        blob = get_bucket().blob(blob_name)
        blob.upload_from_file(io_buf, content_type="image/jpeg", rewind=True)
        
        # Verify upload
        if blob.exists():
            print(f"Successfully uploaded: {blob_name}")
            # Return public URL
            return f"gs://{BUCKET_NAME}/{blob_name}"
        else:
            print(f"Upload verification failed for: {blob_name}")
            return None
    except Exception as e:
        print(f"Error uploading image {image_name}: {str(e)}")
        # traceback.print_exc()
        return None


# Function to compute Intersection over Union (IoU) between two bounding boxes
def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area


# Function to process a frame and detect apples
def detect_objects(frame, frame_count, save=True, detected_boxes=[]):
    if model is None:
        return frame, 0, [], detected_boxes, []

    # Convert BGR to RGB for YOLO input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform detection
    results = model(frame_rgb)

    # Parse results
    detections = results.pandas().xyxy[0]  # Bounding boxes as pandas DataFrame
    apple_count = 0  # Initialize apple count for the current frame

    new_boxes = []  # List to store the bounding boxes of new apples
    current_boxes = []  # List to store all apple boxes in current frame
    cloud_urls = []  # List to store cloud URLs

    for _, detection in detections.iterrows():
        # Get bounding box coordinates
        x_min, y_min, x_max, y_max = map(int,
                                         [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
        confidence = detection['confidence']
        label = detection['name']  # Class label

        if label.lower() == "apple":  # Filter only apples
            current_boxes.append((x_min, y_min, x_max, y_max))
            
            # Check if this apple overlaps with previously detected apples
            is_new_apple = True
            for prev_box in detected_boxes:
                if iou((x_min, y_min, x_max, y_max), prev_box) > IOU_THRESHOLD:
                    is_new_apple = False
                    break

            if is_new_apple:
                apple_count += 1  # Increment apple count
                new_boxes.append((x_min, y_min, x_max, y_max))  # Add new apple to the list

                # Crop the detected apple
                cropped_image = frame[y_min:y_max, x_min:x_max]

                # Save the cropped image locally and to cloud
                if save:
                    # Local save
                    image_name = f"apple_{frame_count}_{x_min}_{y_min}.jpg"
                    output_path = os.path.join(output_dir, image_name)
                    if cropped_image.size > 0:
                        cv2.imwrite(output_path, cropped_image)
                        
                        # Cloud save
                        cloud_url = upload_to_gcs(cropped_image, image_name)
                        if cloud_url:
                            print(f"Added cloud URL for apple: {cloud_url}")
                            cloud_urls.append({
                                "url": cloud_url,
                                "confidence": float(confidence),
                                "coordinates": {
                                    "x_min": int(x_min),
                                    "y_min": int(y_min),
                                    "x_max": int(x_max),
                                    "y_max": int(y_max)
                                }
                            })
                    else:
                        print("Warning: Empty cropped image, skipping save")

            # Draw the bounding box and label on the frame (draw all detected apples)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    return frame, apple_count, new_boxes, current_boxes, cloud_urls


# Save apple count data to JSON file locally and to GCS
def save_apple_count(max_apple_count):
    """Save apple count data to JSON files locally and to Google Cloud Storage"""
    print(f"Saving data with {len(apple_count_data['cloud_image_urls'])} apple images")
    apple_count_data["max_apples_detected"] = max_apple_count
    
    # Save locally
    local_path = "apple_count_data.json"
    with open(local_path, "w") as f:
        json.dump(apple_count_data, f, indent=4)
    print(f"Saved data locally to {local_path}")
    
    # Save to GCS
    if get_bucket():
        try:
            json_data = json.dumps(apple_count_data, indent=4)
            blob_path = f"{session_id}/apple_count_data.json"
            blob = get_bucket().blob(blob_path)
            blob.upload_from_string(json_data, content_type="application/json")
            
            # Verify upload
            if blob.exists():
                print(f"Successfully uploaded data to GCS: {blob_path}")
            else:
                print(f"Failed to verify data upload to GCS: {blob_path}")
        except Exception as e:
            print(f"Error uploading data to GCS: {str(e)}")
            # traceback.print_exc()


# Main function
def main():
    get_bucket() # Ensure bucket is initialized
    
    # Open webcam feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_count = 0
    last_save_time = time.time()
    last_reset_time = time.time()
    max_apples_detected = 0  # Initialize max apple count
    detected_boxes = []  # Store bounding boxes of previously detected apples
    no_detection_count = 0  # Counter for consecutive frames with no apples

    # Upload a frame to indicate session start
    ret, first_frame = cap.read()
    if ret:
        first_frame_url = upload_to_gcs(first_frame, "session_start.jpg")
        if first_frame_url:
            print(f"Uploaded session start frame: {first_frame_url}")
            apple_count_data["session_start_frame"] = first_frame_url

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Get current time
            current_time = time.time()

            # Reset detected boxes periodically to handle scene changes
            if current_time - last_reset_time >= TRACKING_RESET_TIME:
                print("Resetting apple tracking")
                detected_boxes = []
                last_reset_time = current_time

            # Perform detection and save images every 2 seconds
            if current_time - last_save_time >= 2:
                frame, apple_count, new_boxes, current_boxes, cloud_urls = detect_objects(frame, frame_count, save=True, detected_boxes=detected_boxes)
                frame_count += 1
                last_save_time = current_time

                print(f"Frame {frame_count}: Detected {len(current_boxes)} apples, {len(new_boxes)} new")

                # Update maximum apple count if the current frame has more apples
                if len(current_boxes) > max_apples_detected:
                    max_apples_detected = len(current_boxes)
                    print(f"New maximum apple count: {max_apples_detected}")

                # Add the new bounding boxes to the list of detected boxes
                detected_boxes.extend(new_boxes)

                # Store the apple count for this frame
                frame_data = {
                    "frame": frame_count, 
                    "apple_count": len(current_boxes),
                    "timestamp": datetime.now().isoformat()
                }
                apple_count_data["frame_counts"].append(frame_data)
                
                # Store cloud URLs
                if cloud_urls:
                    print(f"Adding {len(cloud_urls)} cloud URLs")
                    apple_count_data["cloud_image_urls"].extend(cloud_urls)
                
                # Count frames with no apples
                if len(current_boxes) == 0:
                    no_detection_count += 1
                else:
                    no_detection_count = 0
            else:
                # Display without saving
                frame, _, _, current_boxes, _ = detect_objects(frame, -1, save=False, detected_boxes=detected_boxes)

            # Display count on frame
            cv2.putText(frame, f"Apples: {len(current_boxes)} | Max: {max_apples_detected}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display the frame with detections
            cv2.imshow('Apple Detection', frame)

            # Break the loop if no apples detected for 10 consecutive processed frames
            if no_detection_count >= 10:
                print("No apples detected for multiple frames. Exiting.")
                break

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User pressed 'q'. Exiting.")
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Save maximum apple count to JSON file before exiting
        save_apple_count(max_apples_detected)
        
        # Upload the last frame
        if 'frame' in locals() and frame is not None:
            last_frame_url = upload_to_gcs(frame, "session_end.jpg")
            if last_frame_url:
                print(f"Uploaded session end frame: {last_frame_url}")
        
        # Save data one more time to ensure everything is captured
        save_apple_count(max_apples_detected)
        
        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Session complete. Detected {len(apple_count_data['cloud_image_urls'])} apples")
        print(f"Data saved to bucket: {BUCKET_NAME}, session: {session_id}")


if __name__ == "__main__":
    main()
