import os
import face_recognition
#import numpy as np
#import cv2
#file = open('train_date.txt','w')
path = 'images'
encodings = []
for filename in os.listdir(path):
	if os.path.splitext(filename)[1] == '.png' or os.path.splitext(filename)[1] == '.jpg':
		image = face_recognition.load_image_file(filename)
		#image = cv2.imread(filename)
		face_encode = face_recognition.face_encodings(image)[0]
		#file.write(face_encode)
		#print(filename)
		#print(filename)
		encodings.append(filename)
print(filename)
#file.close()	
