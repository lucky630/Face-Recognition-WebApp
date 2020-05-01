# import the necessary packages
import face_recognition
from collections import Counter
import pickle
import cv2
from PIL import Image
import os
import sys
import numpy as np

def identify_face():
        Out_fold = r'C:\Users\gurvinder1.singh\Downloads\Facial-Similarity-with-Siamese-Networks-in-Pytorch-master\data\output_fold'

        # load the known faces and embeddings
        print("[INFO] loading encodings...")
        data = pickle.loads(open("encodings.pickle", "rb").read())
        inti = dict(Counter(data["names"]))
        inti['Unknown'] = 1
        print(data["names"])
        
        # load the input image and convert it from BGR to RGB
        ##/home/aiml/ml/share/data/face_recog/examples
        image = cv2.imread(r"C:\Users\gurvinder1.singh\Downloads\Facial-Similarity-with-Siamese-Networks-in-Pytorch-master\data\input_fold\inp.jpg")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         
        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb,model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
         
        # initialize the list of names for each face detected
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
                # attempt to match each face in the input image to our known
                # encodings
                matches = face_recognition.compare_faces(data["encodings"],encoding)
                name = "Unknown"
                final_val = 0
                print(matches)
                # check to see if we have found a match
                if True in matches:
                        # find the indexes of all matched faces then initialize a
                        # dictionary to count the total number of times each face
                        # was matched
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}

                        # loop over the matched indexes and maintain a count for
                        # each recognized face face
                        for i in matchedIdxs:
                                name = data["names"][i]
                                counts[name] = counts.get(name, 0) + 1
                        print(counts)
                        # determine the recognized face with the largest number of
                        # votes (note: in the event of an unlikely tie Python will
                        # select first entry in the dictionary)
                        name = max(counts, key=counts.get)
                        final_val = counts[name]
                confidence_val = int((final_val/inti[name])*100)
                print(confidence_val)
                if confidence_val>70:
                        names.append(name+': '+str(confidence_val))
                else:
                        names.append('Unknown')
                # update the list of names
                #names.append(name)

        ##print(names)
        # loop over the recognized faces

        for ((top, right, bottom, left), name) in zip(boxes, names):
                # draw the predicted face name on the image
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
         
        # show the output image
        result = Image.fromarray(image)
        result.save(os.path.join(Out_fold, 'out.jpg'))
        return names
        print('file saved..')

def main(argv):
    print(identify_face(*argv[1:]))

if __name__ == "__main__":
    import sys
    main(sys.argv)
