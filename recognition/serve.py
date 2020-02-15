import socket
import sys
import numpy as np
import cv2
import imutils
import argparse
import pickle
import os

import track


### INITIALIZE TRACKER WITH ARGUMENTS ###

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

track.conf = args["confidence"]

print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
  "res10_300x300_ssd_iter_140000.caffemodel"])

track.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
track.embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
track.recognizer = pickle.loads(open(args["recognizer"], "rb").read())
track.le = pickle.loads(open(args["le"], "rb").read())


### SETUP SERVER ###

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

      output = track.track_faces(img, dbg=True)

      connection.send(output.encode())

      if not(data):
        print('no more data from', client_address)
        break
      
  finally:
    # Clean up the connection
    connection.close()
