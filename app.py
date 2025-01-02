import logging
from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
try:
    model = load_model('model/facial_expression_model.h5')  # Replace with your model path
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

video_capture = None
def start_video_capture():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            logging.error("Error: Could not open webcam.")
            raise SystemExit("Error: Could not open webcam.")
        logging.info("Webcam started.")
def gen_frames():
    global video_capture
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Failed to capture frame from webcam.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert face to grayscale
            face = face / 255.0  # Normalize the pixel values
            face = np.reshape(face, (1, 48, 48, 1))  # Add batch dimension

            try:
                prediction = model.predict(face)
                max_index = np.argmax(prediction[0])
                predicted_emotion = emotion_labels[max_index]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                logging.error(f"Error during emotion prediction: {e}")
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logging.error("Failed to encode frame.")
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    start_video_capture()  # Start the webcam only when this route is accessed
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        logging.info("Webcam stopped.")
    return "Webcam stopped"

if __name__ == '__main__':
    app.run(debug=True)