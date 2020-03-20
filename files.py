import os, sys
import cv2
import face_recognition
import numpy as np
filenames = []
encodings = []
filepath = 'image'
for filename in os.listdir(filepath):
    if os.path.splitext(filename)[1] == '.png' or os.path.splitext(filename)[1] == '.jpg':
        #image =face_recognition.load_image_file(filename)
        filenames.append(filename)
        image = cv2.imread(filename)
        face_encode = face_recognition.face_encodings(image)[0]
        #file.setdefault(filename,face_encoding)
        encodings.append(face_encode)
        print(filename)
face_locations=[]
face_encodings=[]
face_names=[]
process_this_frame=True
video_capture = cv2.VideoCapture(0)
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, model='cnn')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            #print(best_match_index)

            if matches[best_match_index]:
                name = filenames[best_match_index][:-4]

            #if name == 'Unknown':
                #filename = 'photo'
                #foi_color = rgb_small_frame[]
                #cv2.imwrite(filename+'.jpg',small_frame)
            face_names.append(name)
    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    #cv2.namedWindow('Video',cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Video', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#print(wang_face_landmarks_list)
# Release handle to the webcam

video_capture.release()
cv2.destroyAllWindows()


