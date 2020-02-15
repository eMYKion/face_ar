import cv2
import imutils
import numpy as np
import math

detector = None
conf = None
embedder = None
recognizer = None
le = None

distance_threshold = 70

uid_counter = 0

track_objs = dict()


def box_movement(boxes, dbg):
  global uid_counter

  output_data = ""

  untracked = set()
  for key in track_objs:
    untracked.add(key)

  for (x1b, y1b, x2b, y2b, name_b) in boxes:
    b_ctr_x = (x2b + x1b) / 2
    b_ctr_y = (y2b + y1b) / 2

    tracked = False
    # check if the box is new

    for key in track_objs:
      (x1, y1, x2, y2, name) = track_objs[key]
      ctr_x = (x2 + x1) / 2
      ctr_y = (y2 + y1) / 2

      dist_ctr = math.sqrt((ctr_x - b_ctr_x) ** 2 + (ctr_y - b_ctr_y) ** 2)

      if (dist_ctr < distance_threshold) and (key in untracked):
        untracked.remove(key)
        track_objs[key] = (x1b, y1b, x2b, y2b, name_b)
        output_data += ("POS UPDATE, %d, %d, %d, %d, %d\n" % (key, x1b, y1b, x2b, y2b))
        tracked = True
        break
    
    if not tracked:
      output_data += ("NEW BOX %d, %d, %d, %d, %d\n" % (uid_counter, x1b, y1b, x2b, y2b))
      track_objs[uid_counter] = (x1b, y1b, x2b, y2b, name_b)
      uid_counter += 1
  

  for key in untracked:
    output_data += ("BOX GONE, %d\n" % key)
    del track_objs[key]

  for key in track_objs:
    (_, _, _, _, name) = track_objs[key]
    output_data += ("NAME, %d, %s\n" % (key, name))
  
  output_data += "TERMINATED\n"

  if dbg:
    print(output_data)
  
  return output_data

def process_frame(frame, dbg):
  frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
  frame = imutils.resize(frame, width=600)
  (h, w) = frame.shape[:2]

  print("IMAGE (%d, %d)" % (h, w))

  # construct a blob from the image
  imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(frame, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)

  # apply OpenCV's deep learning-based face detector to localize
  # faces in the input image
  detector.setInput(imageBlob)
  detections = detector.forward()

  boxes = list()

  # loop over the detections
  for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # filter out weak detections
    if confidence > conf:
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
      
      boxes.append((startX, startY, endX, endY, name))

      # draw the bounding box of the face along with the
      # associated probability
      text = "{}: {:.2f}%".format(name, proba * 100)
      y = startY - 10 if startY - 10 > 10 else startY + 10
      cv2.rectangle(frame, (startX, startY), (endX, endY),
        (0, 0, 255), 2)
      cv2.putText(frame, text, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

  if dbg:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
      exit(0)
  
  return box_movement(boxes, dbg)

def track_faces(img, dbg=False):
  return process_frame(img, dbg)
