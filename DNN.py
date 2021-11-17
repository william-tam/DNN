from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from os.path import dirname, join
from os.path import os

# Defining the models needed for DNN face detection framework. 
PROTOTXT_MODEL = 'deploy.prototxt.txt' 
CAFFE_MODEL = 'res10_300x300_ssd_iter_140000_fp16.caffemodel' 

# Reading from the local directory to train the opencv algorithm to detect a face. 
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt' ,'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Begin video capture from the integrated camera. 
Videocap = VideoStream(src=0).start()
time.sleep(2.0)

while True:

	# Continuously reading from the camera so that each frame is captured and resized for face recognition. 
	frame = Videocap.read()
	frame = imutils.resize(frame, width=1000)
 
	# Create a "blob" from the frame. This blob is a collection of images with the same dimensions to be processed by opencv. 
	# (image, scalefactor=1.0, size, mean, swapRB=True)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
 
	net.setInput(blob)
	detections = net.forward()

	# If statement to identify the detections in opencv. 
	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		# If the detection from opncv has a higher "confidence" than 50%, the program will draw a rectangle around the face, with the confidence rating.
		# Confidence: The measurement of the accuracy of face detection made by the program. 
		if confidence < 0.5: 
			continue

		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		text = "{:.2f}%".format(confidence * 100)
		cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
		cv2.putText(frame, text, (startX,(startY-10)),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break

cv2.destroyAllWindows()
Videocap.stop()