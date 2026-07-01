### Project Update:

Based on the feedback received during the project presentation, the model was upgraded from the previous YOLOv8n-cls version to YOLO12n-cls
to improve prediction consistency, especially for video inference.

#### Changes Made
- Re-trained the distracted driver behavior detection model using YOLO12n.
- Updated the trained model (best.pt) with the new YOLO12n weights.
- Evaluated the model on both image and video inputs.
- Verified the frame-wise video prediction summary using the newly trained model.
- Added the project repository with the latest trained model and sample prediction outputs.

#### Updated Results
- Improved prediction consistency during video inference.
- Generated new output images using the YOLO12n model.
- The application now uses the updated YOLO12n model for both image and video behavior classification.

**Technology Stack:** Python, YOLO12n, Ultralytics, OpenCV, Streamlit
#### Limitations

- The reported evaluation metrics are based on the dataset used for training and validation.
- Performance on real-world images and videos may vary due to differences in lighting conditions, camera angles, occlusions, image quality, and other domain shifts.
- Further improvements can be achieved by training on a larger and more diverse dataset, applying data augmentation, and fine-tuning the model on real-world driving scenarios.
