import time
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from picamera2 import Picamera2, Preview
import cv2  # Import OpenCV

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (300, 300)}))  # Adjust size for object detection
picam2.start_preview(Preview.NULL)
picam2.start()

# Load pre-trained Faster R-CNN model with ResNet50 backbone
net = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.COCO_V1')
net.eval()  # Set the model to evaluation mode

# Load class labels for COCO dataset
with open('/home/comp8296/Desktop/Object/Object_Detection_Files/coco.names') as f:
    class_names = [line.strip() for line in f.readlines()]

frame_count = 0
last_logged = time.time()

# Create a window for displaying the camera feed
cv2.namedWindow("Object Detection Feed", cv2.WINDOW_AUTOSIZE)

with torch.no_grad():
    while True:
        # Capture frame using picam2
        frame = picam2.capture_array()

        # Convert frame from RGB to BGR for OpenCV
        image = frame[:, :, [2, 1, 0]]  # Convert from RGB to BGR for OpenCV

        # Ensure the image is a proper NumPy array
        image = np.array(image, dtype=np.uint8).copy()

        # Check the shape and type
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")

        # Convert image to a tensor and normalize
        input_tensor = F.to_tensor(image).unsqueeze(0)  # Add a batch dimension

        # Run the model
        detections = net(input_tensor)

        # Process the output for object detection
        for i in range(len(detections[0]['boxes'])):
            box = detections[0]['boxes'][i].cpu().numpy()  # Ensure box is on CPU and converted to numpy array
            score = detections[0]['scores'][i].cpu().numpy()  # Ensure score is on CPU and converted to numpy array
            if score > 0.5:  # Confidence threshold (can adjust as needed)
                class_id = int(detections[0]['labels'][i].cpu().numpy())  # Ensure label is on CPU and converted to numpy
                (startX, startY, endX, endY) = box.astype("int")

                # Check if the coordinates are within the image dimensions
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(image.shape[1], endX)
                endY = min(image.shape[0], endY)

                # Draw bounding box and label
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"{class_names[class_id]}: {score:.2f}"
                cv2.putText(image, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the camera feed in the window
        cv2.imshow("Object Detection Feed", image)  # Show the processed frame in a window

        # Log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now - last_logged):.2f} fps")
            last_logged = now
            frame_count = 0

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cv2.destroyAllWindows()

