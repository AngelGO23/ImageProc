'''This code is originally from: https://www.youtube.com/watch?v=h56M5iUVgGs&list=LL
I have added personal comments for me to understand better.'''
#Only two libraries
import cv2
import numpy as np

#------------------------------
#Load YOLO
'''The following 2 lines are for setting the neural 
net that we are going to be using. These strings should be the directory to
those files (hence path).'''
cfgPath = 'yolov3-320.cfg'
weightsPath = 'yolov3.weights'

'''We initialize the neural net pasing the two files which are suppose
to contain the structure and the train model.'''
net = cv2.dnn.readNet(weightsPath, cfgPath)
'''Following line is for the file with all the class names the
trained model can detect'''
nameFile = 'coco.names'

'''Following is the opening of the coco.names file, reading it (hence rt), and 
inserting each name into a list called classNames'''
with open(nameFile, 'rt') as f:
    classNames = [line.strip() for line in f.readlines()]

'''The next two lines are for getting the outputLayers from which we
will get the final output image'''
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

#Random colors
# Next line is for having each name class have a different bounding box
colors = np.random.uniform(0, 255, size=(len(classNames), 3))
#------------------------------
#Load Image

img = cv2.imread('homeOffice.jpg')
# fx and fy are dependent on the siz of img
img = cv2.resize(img, None, fx = 1, fy = 1)
height, width, channels = img.shape
#Detecting objects
'''Here we a re beginning to modify the given image so it can 
have the proper metrics for being an input for the neural net.
Blob is going to be a binary representation of the image. 
This makes it easies to be analized by the neural net (also it is the required input).'''
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
'''With the following line, the script extracts the output
from the neural net given the input we gave it in the previous line.
This outs variable CONTAINS the answer to where are the objects in the image.
The for loop a few lines below is for accessing that data structure in the 
correct way to draw the bounding box. Feel free to use the print statement 
to see what the output looks like.'''
outs = net.forward(outputlayers)

'''These lists will contain the index for the boxes,
the confidence it has for the bounding box to identify the correct class name,
and the Ids (the names of the objects that do appear.'''
boxes = []
confidences = []
class_Ids = []
# We begin to analyze the output of the neural net
for out in outs:
    for detec in out:
        scores = detec[5:]
        class_Id = np.argmax(scores)
        confidence = scores[class_Id]
        '''This if statement says that it will only consider the object
        if the class name identified has a confidence greater than 50%'''
        if confidence > 0.5:
            # Obj detected
            # Extract the coordinates for drawing the bounding box
            center_x =  int(detec[0]*width)
            center_y = int(detec[1]*height)
            w = int(detec[2]*width)
            h = int(detec[3]*height)
            #Bounding Box coord
            x = int(center_x -w/2)
            y = int(center_y - h/2)
            # It adds to  the lists the info for drawing the bounding box
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_Ids.append(class_Id)
# Eliminates multiple bounding boxes identifying the same thing
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_ITALIC
for i in range(len(boxes)):
    # We only access the index from the variable indexes declared in one of the previous lines
    if i in indexes:
        # Draw bounding box, class name close to its respective box and chose a cool color (random)
        x, y, w, h = boxes[i]
        label = str(classNames[class_Ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 1, color, 3)


# Finally we show the output
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyWindow('Image')




