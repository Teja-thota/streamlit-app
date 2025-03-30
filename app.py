import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import os

# Ensure YOLOv5 model file exists, otherwise download it
MODEL_PATH = "yolov5s.pt"
if not os.path.exists(MODEL_PATH):
    st.warning("Model file missing. Downloading YOLOv5 model...")
    os.system(f"wget https://github.com/ultralytics/yolov5/releases/download/v6.0/{MODEL_PATH}")

# Load YOLOv5 model with error handling
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    class_names = model.names
except Exception as e:
    st.error(f"Error loading YOLOv5 model: {e}")
    model = None
    class_names = []
    st.stop()  # Stop execution if model fails to load

def detect_objects(image):
    """Detect objects in the image using YOLOv5."""
    if model is None:
        return image, {}
    results = model(image)
    detections = results.xyxy[0]  # Coordinates
    count_objects = {}
    for *box, conf, cls in detections:
        label = class_names[int(cls)]
        count_objects[label] = count_objects.get(label, 0) + 1
    annotated_image = np.array(results.render()[0])
    return annotated_image, count_objects

def live_camera_detection():
    """Perform live camera object detection."""
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    if cap.isOpened():
        while st.session_state.live_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to open camera. Check your camera connection.")
                break
            annotated_frame, counts = detect_objects(frame)
            for i, (obj, count) in enumerate(counts.items()):
                cv2.putText(annotated_frame, f"{obj}: {count}", (10, 30 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)
        cap.release()
        cv2.destroyAllWindows()

# Streamlit UI
st.title("YOLOv5 Object Detection App")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Image Detection", "Live Camera Detection"])

if page == "Home":
    st.header("Welcome to the YOLOv5 Object Detection App! 🎉")
    st.write("""
    - **Image Detection**: Upload an image to detect objects and get their counts.
    - **Live Camera Detection**: Use your webcam for real-time object detection.
    """)
    st.image("icon.jpg", caption="YOLOv5 in Action", use_container_width=True)
elif page == "Image Detection":
    st.header("Image Detection")
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"], key="image_upload")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        annotated_image, counts = detect_objects(image_np)
        st.image(annotated_image, caption="Detected Objects", use_container_width=True)
        st.write("Object Counts:")
        for obj, count in counts.items():
            st.write(f"{obj}: {count}")
elif page == "Live Camera Detection":
    st.header("Live Camera Detection")
    
    if "live_detection" not in st.session_state:
        st.session_state.live_detection = False
    
    start_button = st.button("Start Live Detection")
    stop_button = st.button("Stop Live Detection")
    
    if start_button:
        st.session_state.live_detection = True
    
    if stop_button:
        st.session_state.live_detection = False
    
    if st.session_state.live_detection:
        live_camera_detection()

st.sidebar.info("Powered by YOLOv5 and Streamlit")
