import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

@st.cache_data
def load_model(path):
    return YOLO('best.pt')
    
model = load_model('best.pt')

st.markdown("# Imitating DragonFruitAI Fire Detection")

file  = st.file_uploader('Upload you input file', type=['mp4', 'mov'])
output = st.empty()
if file is not None:
    button = st.button("Predict")

    if button:
        with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_video_path = os.path.join(tmpdirname, 'output_video.mp4')

                    # Read video file
                    tfile = tempfile.NamedTemporaryFile(delete=False) 
                    tfile.write(file.read())

                    cap = cv2.VideoCapture(tfile.name)
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

                    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                    out = cv2.VideoWriter(temp_video_path, fourcc, frame_rate, (frame_width, frame_height))

                    # Progress bar
                    progress_bar = st.progress(0)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    # Process video
                    for i in range(total_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break

                        results = model.predict(frame)

                        # Assuming results[0] is a numpy array (frame with predictions)
                        annotated_frame = results[0].plot()
                        out.write(annotated_frame)
                        output.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_column_width=True)

                        # Update the progress bar
                        progress = int((i + 1) / total_frames * 100)
                        progress_bar.progress(progress)

                    # Release video resources
                    cap.release()
                    out.release()
                    
                    # Display the download button
                    with open(temp_video_path, 'rb') as file:
                        st.download_button(label='Download Annotated Video',
                                        data=file,
                                        file_name='annotated_video.mp4',
                                        mime='video/mp4')

    




