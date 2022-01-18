import face_recognition
import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(0)

known_person=[]
known_image=[]
known_face_encodings=[]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

for file in os.listdir("profiles"):
    try:
        known_person.append(file.replace(".jpg", ""))
        file=os.path.join("profiles/", file)
        known_image = face_recognition.load_image_file(file)
        known_face_encodings.append(face_recognition.face_encodings(known_image)[0])

    except Exception as e:
        pass
    
while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        global name_gui;
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_person[best_match_index]

            face_names.append(name)
    
            name_gui = name

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name_gui, (left + 10, bottom - 10), font, 1.0, (0, 0, 0), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
