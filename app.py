import tkinter as tk
from threading import Thread
from PIL import Image, ImageTk
import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the face detection model
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000 (3).caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the mask detection model
maskNet = load_model("bbest_model_MobileNetV2_with_tuning.h5")

# Placeholder detect_and_predict_mask function
def detect_and_predict_mask(frame, faceNet, maskNet):
    # Replace this with your actual mask detection logic
    # Example code for placeholder function:
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Create a function to start/stop the video stream
def toggle_stream():
    if not hasattr(toggle_stream, "vs"):
        toggle_stream.vs = VideoStream(src=0).start()
        toggle_stream.thread = Thread(target=update_stream)
        toggle_stream.thread.daemon = True
        toggle_stream.thread.start()
        toggle_stream.btn.config(text="Stop Stream")
    else:
        toggle_stream.vs.stop()
        toggle_stream.thread.join()
        toggle_stream.vs = None
        toggle_stream.btn.config(text="Start Stream")

# Function to continuously update the video stream
def update_stream():
    while True:
        if toggle_stream.vs:
            frame = toggle_stream.vs.read()
            frame = imutils.resize(frame, width=400)
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (covering, glasses, plain, sunglasses) = pred

                label_names = ["covering", "glasses", "plain", "sunglasses"]
                label_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

                max_index = np.argmax(pred)
                label = label_names[max_index]
                color = label_colors[max_index]

                label = "{}: {:.2f}%".format(label, pred[max_index] * 100)

                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Display the frame in the Tkinter GUI
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = imutils.resize(img, width=400)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, image=img, anchor=tk.NW)
            canvas.img = img

# Create the main window
root = tk.Tk()
root.title("Mask Detection App")

# Create a button to start/stop the video stream
toggle_stream.btn = tk.Button(root, text="Start Stream", command=toggle_stream)
toggle_stream.btn.pack(pady=10)

# Create a canvas to display the video stream
canvas = tk.Canvas(root, width=400, height=300)
canvas.pack()

# Run the Tkinter main loop
root.mainloop()
