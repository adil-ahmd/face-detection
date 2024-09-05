
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prototxt", required=True,
                        help="Path to Caffe 'deploy' prototxt file")
    parser.add_argument("-m", "--model", required=True,
                        help="Path to Caffe pre-trained model")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="Minimum probability to filter weak detections")
    return vars(parser.parse_args())

def load_model(prototxt, model):
    print("[INFO] Loading model...")
    return cv2.dnn.readNetFromCaffe(prototxt, model)

def initialize_video_stream():
    print("[INFO] Starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    return vs

def detect_faces(frame, net, confidence_threshold):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence >= confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = f"{confidence * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return frame

def main():
    args = parse_arguments()
    net = load_model(args["prototxt"], args["model"])
    vs = initialize_video_stream()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=720)

        frame = detect_faces(frame, net, args["confidence"])

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
