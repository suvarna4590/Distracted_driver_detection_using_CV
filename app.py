import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
from collections import Counter

st.set_page_config(
    page_title="Distracted Driver Behavior Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Distracted Driver Behavior Detection")
st.write("Upload an image or video to classify distracted driver behavior.")

model = YOLO(r"best.pt")

class_names = {
    "c0": "safe driving",
    "c1": "texting - right",
    "c2": "talking on the phone - right",
    "c3": "texting - left",
    "c4": "talking on the phone - left",
    "c5": "operating the radio",
    "c6": "drinking",
    "c7": "reaching behind",
    "c8": "hair and makeup",
    "c9": "talking to passenger"
}

option = st.radio("Choose input type", ["Image", "Video"])

if option == "Image":
    uploaded_image = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict Image"):
            results = model.predict(image, imgsz=224)
            result = results[0]

            predicted_class_id = result.probs.top1
            confidence = result.probs.top1conf.item()

            predicted_folder_name = model.names[predicted_class_id]
            readable_name = class_names[predicted_folder_name]

            st.success(f"Prediction: {readable_name}")
            st.info(f"Confidence: {confidence * 100:.2f}%")

elif option == "Video":
    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        st.video(temp_file.name)

        if st.button("Predict Video"):
            cap = cv2.VideoCapture(temp_file.name)

            predictions = []

            st.write("Processing video frames...")

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run prediction on EVERY frame
                results = model.predict(frame_rgb, imgsz=224, verbose=False)
                result = results[0]

                predicted_class_id = result.probs.top1
                predicted_folder_name = model.names[predicted_class_id]
                readable_name = class_names[predicted_folder_name]

                predictions.append(readable_name)

            cap.release()

            if predictions:
                final_prediction = Counter(predictions).most_common(1)[0][0]

                st.success(f"Final Video Prediction: {final_prediction}")

                st.subheader("Frame Prediction Summary")
                prediction_counts = Counter(predictions)

                for behavior, count in prediction_counts.items():
                    st.write(f"{behavior}: {count} frames")
            else:
                st.error("No frames were processed.")