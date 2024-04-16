import cv2
import csv
import os
import face_recognition
import datetime

import numpy


# CODE FOR FACE DETECTION & RECOGNIZATION

#Load known faces and their names from the "faces " folder
known_faces = [] #array for faces
known_names=[]   #array for names

#to take images from 'faces' folder & encode them 1 by 1 then append in the ' known_faces =[] ' list
for filename in os.listdir(r"E:\\WorksSpace\\Clg_Project\\FaceRecognitonSystem\\faces"):
    image = face_recognition.load_image_file(os.path.join(r"E:\\WorksSpace\\Clg_Project\\FaceRecognitonSystem\\faces", filename)) # images under faces floder will store in ' image ' folder according to file name
    encoding = face_recognition.face_encodings(image)[0] # here image is taken & and the value of the 0th index is filled in this ' encoding ' variable
    known_faces.append(encoding)  # here encoding variable will be append in 'known_faces= []'
    known_names.append(os.path.splitext(filename)[0])

video_capture = cv2.VideoCapture(0) #here camera is accessed 0=laptops own camera, 1 =camera attached to usb port 1, 2= camera attached to usb port 2

attendance_marked = False #it is used to don't give attendace nmore than 1 time in a date

while True:
    ret, frame = video_capture.read()  # ret( it is used to check camera is capturing or not) & frame are 2 variables  # this is used to turn on camera and capture frame by frame and store

    rgb_frame = numpy.ascontiguousarray(frame[:,:,::-1]) # [:,:,:: -1] is string slicing concept in python.

    face_locations = face_recognition.face_locations(rgb_frame) # in web cam frame is there any person (1,2,4,5,100) their face will be store in this variable
    face_encodings = face_recognition.face_encodings(rgb_frame,face_locations) # after getting face_location, compare encoding with face loction so this tep is used.

    recognized_names =[] #empty list

    for face_encoding in face_encodings:
        matches= face_recognition.compare_faces(known_faces,face_encoding) #here we compare known faces with face encoding to check they are mathing or not
        name = 'Unknown'

        if True in matches:
            matched_indices= [i for i,match in enumerate(matches) if match] # this is called "List Comprihension concept in python" (Assuming of the lists) , it is used write the  code in one line
            for index in matched_indices:
                name = known_names[index]
                recognized_names.append(name) # here saved known faces which are matched are appened in recognized_names list



    # CODE FOR GIVING ATTANDANCE


    if len(recognized_names)>0:
        current_time = datetime.datetime.now().strftime('%H:%M:%S') #here the current time of the face recognition store in  'current_time' variable in the format of (hour : minute : second) using "srtftime('%H:%M:%S') " method
        with open('attendance.csv','r') as file: # here csv file of attendance is opened and checked how much row  & coloumns are and check the attandance on read mode using 'r'
            reader = csv.reader(file)  # csv file opend as a read mode in 'reader ' variable
            existing_names = set(row[0] for row in reader if row)  # This will skip empty lists
        with open('attendance.csv','a' , newline='') as file: # open attendance.csv to write (name and time)
            writer = csv.writer(file)
            for name in recognized_names:
                if name not in existing_names:
                    writer.writerow([name , current_time])
                    existing_names.add(name)

        attendance_marked = True

    cv2.imshow('camera',frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or attendance_marked:
        break

video_capture.release()
cv2.destroyAllWindows()