import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from collections import Counter
from ultralytics import YOLO

st.set_page_config(page_title="Vehicle Detection & Counting", page_icon="🚗", layout="wide")

st.title("🚗 Vehicle Detection & Counting System")
st.write("Upload a road or traffic image to detect vehicles and count each vehicle type.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 0.95, 0.40, 0.05)

uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    with st.spinner("Detecting vehicles..."):
        results = model.predict(source=img_array, conf=conf_threshold, verbose=False)

        for res in results:
            class_ids = res.boxes.cls.int().tolist()
            detected_names = [model.names[cls_id] for cls_id in class_ids]
            vehicle_counts = dict(Counter(detected_names))
            total_vehicles = sum(vehicle_counts.values())

            annotated_bgr = res.plot()
            annotated_rgb = annotated_bgr[:, :, ::-1]
            break

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detection Result")
        st.image(annotated_rgb, use_container_width=True)

    st.divider()
    st.subheader("📊 Vehicle Count Analysis")
    st.metric("Total Vehicles Detected", total_vehicles)

    if vehicle_counts:
        summary_text = ", ".join([f"**{k}**: {v}" for k, v in vehicle_counts.items()])
        st.success(f"Detected Vehicles → {summary_text}")

        df_counts = pd.DataFrame(
            list(vehicle_counts.items()),
            columns=["Vehicle Type", "Count"]
        )
        st.table(df_counts)
    else:
        st.warning("No vehicles detected. Try lowering the Confidence Threshold in the sidebar.")
