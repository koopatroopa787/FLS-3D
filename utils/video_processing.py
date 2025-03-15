import os
import cv2

def extract_frames(video_file, downsample_rate=8, output_folder='frames'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % downsample_rate == 0:
            frame_filename = os.path.join(output_folder, f'frame_{saved_frame_count:05d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames and saved them to {output_folder}.")
