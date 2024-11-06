import ast
import os

from matplotlib import pyplot as plt

os.makedirs('./Results', exist_ok=True)
os.makedirs('./Pictorial Results', exist_ok=True)
os.makedirs('./Saved data', exist_ok=True)
import cv2
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from save_load import save, load
from MobileAnchorNet import mobile_anchor_net
from PIL import Image
import IPython.display as display
from feature_extraction import feature_extraction
from sklearn.model_selection import train_test_split


def datagen():

    BaseDir = './Dataset/DATS_2022/Images and XML files/XML images/'
    xml_Dir = os.listdir(BaseDir)
    data = []

    for xmlFile in xml_Dir:
        xml_path = os.path.join(BaseDir, xmlFile)

        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract information from XML elements and create a dictionary for each image
        image_data = {
            "filename": root.find("filename").text,
            "objects": []
        }

        for obj in root.findall("object"):
            object_data = {
                "object_name": obj.find("name").text,
                "xmin": int(obj.find("bndbox/xmin").text),
                "ymin": int(obj.find("bndbox/ymin").text),
                "xmax": int(obj.find("bndbox/xmax").text),
                "ymax": int(obj.find("bndbox/ymax").text),
            }
            image_data["objects"].append(object_data)

        filename = root.find("filename").text
        image = cv2.imread('./Dataset/DATS_2022/Images and XML files/' + filename)
        if image is None:
            continue

        # Define color ranges for yellow and white lanes
        lower_yellow = np.array([102, 115, 123], dtype=np.uint8)
        upper_yellow = np.array([120, 133, 125], dtype=np.uint8)

        lower_white = np.array([109, 121, 125], dtype=np.uint8)
        upper_white = np.array([128, 136, 135], dtype=np.uint8)

        # Create masks for yellow and white lanes
        mask_yellow = cv2.inRange(image, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(image, lower_white, upper_white)

        # Combine masks
        mask_combined = cv2.bitwise_or(mask_yellow, mask_white)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Add color thresholding to the grayscale image
        _, thresh_color = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        # Combine the color and grayscale masks
        mask_combined = cv2.bitwise_or(mask_combined, thresh_color)

        # Apply Canny edge detection
        edges = cv2.Canny(mask_combined, 50, 150)

        # Define a region of interest (ROI) mask
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[(100, height), (width // 2 - 50, height // 2 + 50),
                             (width // 2 + 50, height // 2 + 50), (width - 50, height)]], np.int32)
        cv2.fillPoly(mask, polygon, 255)

        # Apply the ROI mask
        masked_edges = cv2.bitwise_and(edges, mask)

        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

        # Create a copy of the image to draw the lanes
        lanes_image = np.copy(image)

        # List to store bounding boxes
        bounding_boxes = []

        # Draw detected lanes and bounding boxes
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Filter lines based on slope
                slope = (y2 - y1) / (x2 - x1 + 1e-10)  # Adding a small value to avoid division by zero
                if abs(slope) < 0.5:  # Adjust the slope threshold as needed
                    continue

                # Lane width constraint
                lane_width = x2 - x1
                if lane_width < 50 or lane_width > 300:  # Adjust the width thresholds as needed
                    continue

                # Draw the detected lane
                cv2.line(lanes_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

                # Calculate bounding box coordinates
                xmin = min(x1, x2)
                xmax = max(x1, x2)
                ymin = min(y1, y2)
                ymax = max(y1, y2)

                # Draw bounding box

                cv2.rectangle(lanes_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                # Append bounding box coordinates to the list
                bounding_boxes.append(((xmin, ymin), (xmax, ymax)))

        if bounding_boxes:

            xmin = min([bounding_box[0][0] for bounding_box in bounding_boxes])
            ymin = min([bounding_box[0][1] for bounding_box in bounding_boxes])
            xmax = max([bounding_box[1][0] for bounding_box in bounding_boxes])
            ymax = max([bounding_box[1][1] for bounding_box in bounding_boxes])

            lane_element = root.find("lane")
            if lane_element is None:
                lane_data = {
                    "object_name": 'lane',
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }

                image_data["objects"].append(lane_data)
        else:
            continue

        # Append the image_data to the data list
        data.append(image_data)

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    df.to_csv('Dataset.csv', index=False)



    data = pd.read_csv('Dataset.csv')
    features = []
    labels = []

    imgResult = './Pictorial Results/'

    for index, row in data.iterrows():
        image_file = row['filename']
        image_path = './Dataset/DATS_2022/Images and XML files/' + image_file
        image = cv2.imread(image_path)
        if image is None:
            continue
        resieImage = cv2.resize(image, (2240, 2240), interpolation=cv2.INTER_AREA)
        cv2.imwrite(imgResult+'1. Original Image.png', image)

        # preprocessing
        # Noise Reduction - Bilateral Filter
        DenoisedImg = cv2.bilateralFilter(resieImage, 5, 50, 50)
        cv2.imwrite(imgResult + '2. Denoise Image.png', DenoisedImg)

        # Image Enhancement - Gamma Correction
        # Apply gamma correction.
        gamma = 0.8
        gamma_corrected = np.array(255 * (DenoisedImg / 255) ** gamma, dtype='uint8')
        cv2.imwrite(imgResult + '3. Enhanced Image.png', gamma_corrected)

        # Image Normalization

        normalized_image = cv2.normalize(gamma_corrected, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        cv2.imwrite(imgResult + '4. Normalized image.png', normalized_image)

        object_names = ast.literal_eval(row['objects'])
        save('proposed.names', object_names)

        img = normalized_image.reshape(1, normalized_image.shape[0], normalized_image.shape[1], normalized_image.shape[2])

        # Segmentation
        segmented_image, seg_images, objects = mobile_anchor_net(img, object_names, image)
        for imgs in seg_images:
            imgs = cv2.resize(imgs, (256, 256))
            feature = feature_extraction(imgs)
            feature = feature / np.max(feature)
            feature = abs(feature)
            feature = np.nan_to_num(feature)
            features.append(feature)
        labels.extend(objects)
        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(segmented_image)
        # Display the image with bounding boxes and names
        display.display(pil_image)
        pil_image.save('./Pictorial Results/5. Object Detected Image.png')

    features = np.array(features)
    unique_values = list(set(labels))
    label_mapping = {label: idx for idx, label in enumerate(unique_values)}
    # Convert original list to numerical labels
    numerical_labels = [label_mapping[label] for label in labels]
    numerical_labels = np.array(numerical_labels)
    save('numerical labels', numerical_labels)

    learning_rate = [0.7, 0.8]  # train size
    for learn_rate in learning_rate:
        x_train, x_test, y_train, y_test = train_test_split(features, numerical_labels, train_size=learn_rate)
        save('x_train_' + str(int(learn_rate * 100)), x_train)
        save('x_test_' + str(int(learn_rate * 100)), x_test)
        save('y_train_' + str(int(learn_rate * 100)), y_train)
        save('y_test_' + str(int(learn_rate * 100)), y_test)
    save('features', features)
    save('labels', labels)



'''


cap = cv2.VideoCapture('./4.mp4')
processed_frames = []
while True:
    ret, image = cap.read()
    if not ret:
        break

    frame = cv2.resize(image, dsize=(500, 500), interpolation=cv2.INTER_AREA)

    cv2.imwrite('1. Original Image.png', frame)

    # preprocessing
    # Noise Reduction - Bilateral Filter
    DenoisedImg = cv2.bilateralFilter(frame, 5, 50, 50)
    cv2.imwrite('2. Denoise Image.png', DenoisedImg)

    # Image Enhancement - Gamma Correction
    # Apply gamma correction.
    gamma = 0.8
    gamma_corrected = np.array(255 * (DenoisedImg / 255) ** gamma, dtype='uint8')
    cv2.imwrite('3. Enhanced Image.png', gamma_corrected)

    # Image Normalization

    normalized_image = cv2.normalize(gamma_corrected, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite('4. Normalized image.png', normalized_image)

    net = cv2.dnn.readNet('proposed.cfg', 'proposed.weights')
    layer_names = net.getUnconnectedOutLayersNames()

    # Load class names
    with open('proposed.names', 'r') as f:
        classes = f.read().strip().split('\n')

    # Resize and normalize the image
    blob = cv2.dnn.blobFromImage(normalized_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    outs = net.forward(layer_names)

    # Process the results
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # You can adjust this threshold
                # Get bounding box coordinates
                center_x, center_y, w, h = (detection[0:4] * np.array(
                    [normalized_image.shape[1], normalized_image.shape[0], normalized_image.shape[1], normalized_image.shape[0]])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                # Draw bounding box
                cv2.rectangle(normalized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Get class name
                class_name = classes[class_id]

                # Add class label
                label = class_name
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(normalized_image, label, (x, y - 10), font, 0.2, (255, 0, 0), 1, cv2.LINE_AA)
    processed_frames.append(normalized_image.copy())
    if len(processed_frames) == 1500:
        break

# Release the video capture object
cap.release()

# Create a VideoWriter object
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')

height, width, _ = processed_frames[0].shape
fps = 30  # Adjust the frames per second as needed
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write the processed frames to the video
for frame in processed_frames:
    video_writer.write(frame)

# Release the VideoWriter object
video_writer.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()


'''
import cv2
import numpy as np

# Function to detect lanes in a frame
def detect_lanes(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    label = 'lane'
    # Draw the detected lines on a blank canvas
    lane_frame = np.zeros_like(frame)
    if lines is not None:
        x1 = min([bounding_box[0][0] for bounding_box in lines])
        y1 = min([bounding_box[0][1] for bounding_box in lines])
        x2 = max([bounding_box[0][2] for bounding_box in lines])
        y2 = max([bounding_box[0][3] for bounding_box in lines])
        cv2.line(lane_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, label, (x1, y1 - 10), font, 0.2, (255, 0, 0), 1, cv2.LINE_AA)

    return lane_frame

# Open the video file
cap = cv2.VideoCapture('./5.mp4')

# Load YOLO
net = cv2.dnn.readNet('proposed.cfg', 'proposed.weights')
layer_names = net.getUnconnectedOutLayersNames()

# Load class names
with open('proposed.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Initialize a list to store processed frames
processed_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for processing (if needed)
    frame = cv2.resize(frame, dsize=(500, 500), interpolation=cv2.INTER_AREA)

    # Detect lanes in the frame
    lane_frame = detect_lanes(frame)
    # plt.imshow(lane_frame)
    # plt.show()

    # Resize and normalize the frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run YOLO forward pass
    outs = net.forward(layer_names)

    # Process the results for object detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # You can adjust this threshold
                # Get bounding box coordinates
                center_x, center_y, w, h = (detection[0:4] * np.array(
                    [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                # Draw bounding box for objects
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

                # Get class name
                class_name = classes[class_id]

                # Add class label for objects
                label = class_name
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, label, (x, y - 10), font, 0.2, (255, 0, 0), 1, cv2.LINE_AA)

    # Overlay lane detection results onto the frame
    result_frame = cv2.addWeighted(frame, 1, lane_frame, 0.5, 0)
    # Append the processed frame to the list
    processed_frames.append(result_frame.copy())



# Release the video capture object
cap.release()

# Create a VideoWriter object
output_video_path = 'output_object_lane_detection.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30  # Adjust the frames per second as needed
height, width, _ = processed_frames[0].shape
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Write the processed frames to the video
for frame in processed_frames:
    video_writer.write(frame)

# Release the VideoWriter object
video_writer.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
