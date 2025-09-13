import tkinter as tk
from tkinter import filedialog, messagebox, font
import cv2
from collections import deque
import numpy as np
import os
import tensorflow
from tensorflow.keras.models import load_model

# Global constants and configurations
SEQUENCE_LENGTH = 20  # Number of frames per sequence
CLASSES_LIST = ["Clapping", "Meet and Split", "Sitting", "Standing Still", "Walking", "Walking While Reading Book", "Walking While Using Phone"]

MODEL_PATH = 'D:\\min2_activity\\activity_recognition_lstm.h5'

# Load the trained model
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# Initialize MediaPipe Pose
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def predict_activity(video_path, output_path, model, sequence_length=SEQUENCE_LENGTH):
    """ Function to track and annotate detected person in a video """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
    frames_queue = deque(maxlen=sequence_length)
    predicted_class_name = ''

    while cap.isOpened():
        ret, frame = cap.read()

        # Check if the frame is None or ret is False
        if not ret or frame is None:
            print("Error: Failed to capture frame.")
            break

        # Pose estimation for the current frame
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        except cv2.error as e:
            print(f"Error converting frame to RGB: {e}")
            continue  # Skip the invalid frame

        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Draw pose landmarks and bounding box
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            frames_queue.append(keypoints)

            # Calculate bounding box
            x_min = min([landmark.x for landmark in results.pose_landmarks.landmark])
            x_max = max([landmark.x for landmark in results.pose_landmarks.landmark])
            y_min = min([landmark.y for landmark in results.pose_landmarks.landmark])
            y_max = max([landmark.y for landmark in results.pose_landmarks.landmark])

            # Scale to image coordinates
            x_min = int(x_min * width)
            x_max = int(x_max * width)
            y_min = int(y_min * height)
            y_max = int(y_max * height)

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Only predict if we have enough frames in the queue
            if len(frames_queue) == sequence_length:
                predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
                predicted_label = np.argmax(predicted_labels_probabilities)
                predicted_class_name = CLASSES_LIST[predicted_label]

        # Overlay the action label
        cv2.putText(frame, predicted_class_name, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved at {output_path}")


# GUI Class
class ActivityRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Activity Recognition App")
        self.root.geometry("800x600")  # Adjust size for sidebar and main content
        self.root.configure(bg="#f0f8ff")  # Light blue background

        # Define the video path and output path
        self.video_path = None
        self.output_path = "output_video.mp4"

        # Create a custom font
        self.custom_font = font.Font(family="Helvetica", size=12, weight="bold")

        # Create a frame for the sidebar
        self.sidebar = tk.Frame(self.root, width=200, bg="#87cefa", relief="sunken")
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")

        # Create 8 buttons for the activity classes in the sidebar
        self.class_buttons = []
        for i, class_name in enumerate(CLASSES_LIST):
            btn = tk.Button(self.sidebar, text=class_name, command=lambda i=i: self.select_video_for_class(i), 
                            bg="#87cefa", font=self.custom_font)
            btn.grid(row=i, column=0, pady=5, sticky="ew")
            self.class_buttons.append(btn)
        # Create buttons for video processing in the main area
        self.process_button = tk.Button(self.sidebar, text="🎬 Process Video", command=self.process_video, state=tk.DISABLED, bg="#87cefa", font=self.custom_font)
        self.process_button.grid(row=8, column=0, pady=20)

        self.clear_button = tk.Button(self.sidebar, text="🗑️ Clear", command=self.clear_selection, state=tk.DISABLED, bg="#ff6347", font=self.custom_font)
        self.clear_button.grid(row=9, column=0, pady=20)

        self.webcam_button = tk.Button(self.sidebar, text="📷 Start Webcam", command=self.start_webcam, bg="#87cefa", font=self.custom_font)
        self.webcam_button.grid(row=10, column=0, pady=20)

        # Create a frame for the main content (video processing area)
        self.main_area = tk.Frame(self.root, bg="#f0f8ff")
        self.main_area.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Add a header label
        self.header = tk.Label(self.main_area, text="Human Activity Recognition System!", bg="#f0f8ff", font=font.Font(size=16, weight="bold"))
        self.header.grid(row=4, column=0, pady=10)


        self.label = tk.Label(self.main_area, text="No video selected", bg="#f0f8ff", font=self.custom_font)
        self.label.grid(row=3, column=0, pady=20)

        
        # Create a canvas to display video output
        self.canvas = tk.Canvas(self.main_area, width=500, height=500, bg="black")
        self.canvas.grid(row=5, column=0, pady=20)

    def select_video_for_class(self, class_index):
        """Function to allow user to select a video based on the class button clicked."""
        class_name = CLASSES_LIST[class_index]
        self.video_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")])
        if self.video_path:
            self.label.config(text=f"Selected video for: {class_name}")
            self.process_button.config(state=tk.NORMAL)  # Enable process button
            self.clear_button.config(state=tk.NORMAL)   # Enable clear button

    def clear_selection(self):
        """Function to clear the selected video and reset the GUI."""
        self.video_path = None
        self.label.config(text="No video selected")
        self.process_button.config(state=tk.DISABLED)  # Disable process button
        self.clear_button.config(state=tk.DISABLED)    # Disable clear button

    def process_video(self):
        """Function to process the selected video and display the output in the canvas."""
        if not self.video_path:
            messagebox.showerror("Error", "No video selected! ")
            return

        # Process the video using the predict_activity function
        output_path = "output_video.mp4"
        predict_activity(self.video_path, output_path, model)

        # Display the output video in the canvas
        self.display_video(output_path)

    def display_video(self, video_path):
        """Function to display the processed output video in the canvas."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Failed to open output video. ")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to fit within the canvas size
            frame_resized = cv2.resize(frame, (500, 500))

            # Convert the frame to a format suitable for tkinter canvas
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)

            # Update the canvas with the frame
            self.canvas.create_image(0, 0, anchor="nw", image=frame_tk)
            self.root.update_idletasks()  # Refresh the canvas to display the new frame

            # Exit loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    def start_webcam(self):
        """Function to process activity recognition from the webcam."""
        self.process_webcam()  # Call the webcam processing function

    def process_webcam(self):
        """Function to predict activity from webcam feed."""
        cap = cv2.VideoCapture(0)  # Use the default webcam

        if not cap.isOpened():
            print("Error: Could not access webcam.")
            return

        frames_queue = deque(maxlen=SEQUENCE_LENGTH)
        predicted_class_name = ''
        
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret or frame is None:
                print("Error: Failed to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Draw pose landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
                frames_queue.append(keypoints)

                if len(frames_queue) == SEQUENCE_LENGTH:
                    predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
                    predicted_label = np.argmax(predicted_labels_probabilities)
                    predicted_class_name = CLASSES_LIST[predicted_label]

            # Display the predicted activity
            cv2.putText(frame, predicted_class_name, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

            # Resize the frame to fit the canvas
            frame_resized = cv2.resize(frame, (500, 500))

            # Convert frame for tkinter
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)

            # Update the canvas with the webcam frame
            self.canvas.create_image(0, 0, anchor="nw", image=frame_tk)
            self.root.update_idletasks()

            # Exit loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()


if __name__ == "__main__":
    from PIL import Image, ImageTk
    root = tk.Tk()
    app = ActivityRecognitionApp(root)
    root.mainloop()
