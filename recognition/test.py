import socket
import sys
import numpy as np
import cv2
import imutils
import argparse
import pickle
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
      cv2.resize(frame, (300, 300)), 1.0, (300, 300),
      (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
      # extract the confidence (i.e., probability) associated with
      # the prediction
      confidence = detections[0, 0, i, 2]

      # filter out weak detections
      if confidence > args["confidence"]:
          # compute the (x, y)-coordinates of the bounding box for
          # the face
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")

          # extract the face ROI
          face = frame[startY:endY, startX:endX]
          (fH, fW) = face.shape[:2]

          # ensure the face width and height are sufficiently large
          if fW < 20 or fH < 20:
              continue

          # construct a blob for the face ROI, then pass the blob
          # through our face embedding model to obtain the 128-d
          # quantification of the face
          faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False)
          embedder.setInput(faceBlob)
          vec = embedder.forward()

          # perform classification to recognize the face
          preds = recognizer.predict_proba(vec)[0]
          j = np.argmax(preds)
          proba = preds[j]
          name = le.classes_[j]

          # draw the bounding box of the face along with the
          # associated probability
          text = "{}: {:.2f}%".format(name, proba * 100)
          y = startY - 10 if startY - 10 > 10 else startY + 10
          cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
          cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        exit(0)




# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('', 8000)
print('starting up on %s port %s' % server_address)
sock.bind(server_address)
# Listen for incoming connections
sock.listen(1)

real_h, real_w = 1080, 2048

scaling_factor = 4

chunk_size = real_h * real_w // (scaling_factor ** 2)

while True:
    # Wait for a connection
    print('waiting for a connection')
    connection, client_address = sock.accept()
    try:
      print('connection from', client_address)
      
      # Receive the data in small chunks and retransmit it
      while True:
          received = 0
          while received < chunk_size:
              data = connection.recv(chunk_size, socket.MSG_WAITALL)
              received += len(data)
          # print(received)

          img = np.frombuffer(data, dtype=np.ubyte).reshape((real_h // scaling_factor, 
                              real_w // scaling_factor))
          # print(img / 255)

          process_frame(img)
          # cv2.imshow('test', img / 255)
          # if(cv2.waitKey(1) == 27):
          #    break

          # print(data.encode("hex"))
          if not(data):
              print('no more data from', client_address)
              break
            
    finally:
        # Clean up the connection
        connection.close()