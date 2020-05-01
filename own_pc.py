import socket
import pygame
from PIL import Image
import numpy as np
import cv2
import face_recognition
import pickle
import time
from collections import Counter
import imutils

class Vidcamera1(object):
    def __init__(self):
        print("[INFO] loading encodings...")
        self.data11 = pickle.loads(open('encodings_hog.pickle', "rb").read())
        self.inti = dict(Counter(self.data11["names"]))
        self.inti['Unknown'] = 1
        self.timer = 0
        self.previousImage = ""
        self.image = ""
        self.clock = pygame.time.Clock()
        self.video = cv2.VideoCapture(0)
        print(type(self.data11["encodings"]))

    ## processing the frame.
    def process_frame(self,frame1):
        frame=frame1
        #print(type(frame))
        final_val=0
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        #print('1')
        face_names = []
        for face_encoding in face_encodings:
            #print(type(data11))
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.data11["encodings"], face_encoding)
            #print(matches)
            name = "Unknown"
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = self.data11["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                print(counts)
                name = max(counts, key=counts.get)
                final_val = counts[name]
            confidence_val = int((final_val/self.inti[name])*100)
            print(confidence_val)
            #print(face_locations)
            if confidence_val>78:
                face_names.append(name+': '+str(confidence_val))
            else:
                face_names.append('Unknown1 : '+str(confidence_val))
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            #cv2.imwrite('01.jpg',frame)
        return frame

    #Main program loop:
    def framing(self):
          if self.timer < 1:
            success, data = self.video.read()
            frame = self.process_frame(data)
            #print('recog_back')
            #Set the timer back to 30:
            self.timer = 0
          else:
            #Count down the timer:
            self.timer -= 1
          #We store the previous recieved image incase the client fails to recive all of the data for the new image:
          self.previousImage = self.image
          try:
            self.image = frame
          except:
            #If we failed to recieve a new image we display the last image we revieved:
            self.image = self.previousImage
          #Set the var output to our image:    
          output = self.image
          #print('frame')
          #We set our clock to tick 60 times a second, which limits the frame rate to that amount:
          self.clock.tick(1000)
          #pygame.display.flip()
          ret, jpeg = cv2.imencode('.jpg', output)
          return jpeg.tobytes()
