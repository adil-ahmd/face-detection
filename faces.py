import numpy as np
import argparse
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser(description="Detect faces in an image using a pre-trained Caffe model.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image.")
    parser.add_argument("-p", "--prototxt", required=True, help="Path to the Caffe 'deploy' prototxt file.")
    parser.add_argument("-m", "--model", required=True, help="Path to the Caffe pre-trained model.")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections.")
    return vars(parser.parse_args())

def load_model(prototxt, model):
    print("[INFO] Loading model...")
    return cv2.dnn.readNetFromCaffe(prototxt, model)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    return image, blob, h, w

def detect_faces(net, blob):
    print("[INFO] Computing object detections...")
    net.setInput(blob)
    return net.forward()

def draw_detections(image, detections, confidence_threshold, h, w):
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            draw_bounding_box(image, startX, startY, endX, endY, confidence)

def draw_bounding_box(image, startX, startY, endX, endY, confidence):
    text = f"{confidence * 100:.2f}%"
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

def main():
    args = parse_arguments()
    
    net = load_model(args["prototxt"], args["model"])
    
    image, blob, h, w = preprocess_image(args["image"])
    
    detections = detect_faces(net, blob)
    
    draw_detections(image, detections, args["confidence"], h, w)
    
    cv2.imshow("Output", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
