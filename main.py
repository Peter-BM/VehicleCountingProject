import streamlit as st
import cv2
from ultralytics import YOLO, solutions
import tempfile
import numpy as np


# Load YOLOv8 OpenVINO model
@st.cache_resource
def load_model():
    return YOLO("yolov8s_openvino_model/")


model = load_model()

# Streamlit Sidebar settings for class selection and line adjustment
with st.sidebar.title("Object Detection Settings"):
    with st.form("parameters"):
        classes = ['Car', 'Motorcycle', 'Bus', 'Truck', 'Person']
        class_indices = {'Car': 2, 'Motorcycle': 3, 'Bus': 5, 'Truck': 7, 'Person': 0}
        selected_classes = st.multiselect('Select Classes to Detect', classes, default=classes)
        classes_to_count = [class_indices[cls] for cls in selected_classes]

        st.sidebar.title("Line Settings")
        line_y = st.slider("Line Y-Position", min_value=0, max_value=720, value=550)

        line_points = [(100, line_y), (1300, line_y)]

        # Video upload
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])

        # Submit button to start processing
        submit = st.form_submit_button("Submit", type="primary")

# Ensure the video processing only happens after form submission
if submit and uploaded_file is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Init Object Counter
    counter = solutions.ObjectCounter(
        view_img=False,
        reg_pts=line_points,
        names=model.names,
        draw_tracks=False,
        line_thickness=2,
    )

    stframe = st.empty()  # Placeholder to display video

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break

        # Run model
        tracks = model.track(im0, persist=True, classes=classes_to_count)

        # Perform counting
        im0 = counter.start_counting(im0, tracks)

        # Draw the line
        cv2.line(im0, line_points[0], line_points[1], (0, 255, 0), 2)

        # Convert BGR to RGB for Streamlit display
        im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

        # Update frame in Streamlit
        stframe.image(im0_rgb, channels="RGB", use_column_width=True)

        # Stop video processing if the user clicks the stop button
        if st.sidebar.button("Stop Video"):
            break

    cap.release()
