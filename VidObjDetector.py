import cv2
import numpy as np
import time
#------------------------------
#Load YOLO
cfgPath = 'yolov3-320.cfg'
weightsPath = 'yolov3.weights'

net = cv2.dnn.readNet(weightsPath, cfgPath)
nameFile = 'coco.names'

with open(nameFile, 'rt') as f:
    classNames = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

#Random colors
colors = np.random.uniform(0, 255, size=(len(classNames), 3))
#------------------------------
#Load Image

cap = cv2.VideoCapture(0)
timeNow = time.time()
frame = 0
font = cv2.FONT_ITALIC
while True:
    _, img = cap.read()
    frame +=1
    height, width, channels = img.shape
    #Detecting objects

    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    #comment
    outs = net.forward(outputlayers)

    #Show information on screen
    boxes = []
    confidences = []
    class_Ids = []
    for out in outs:
        for detec in out:
            scores = detec[5:]
            class_Id = np.argmax(scores)
            confidence = scores[class_Id]
            if confidence > 0.5:
                # Obj detected
                center_x =  int(detec[0]*width)
                center_y = int(detec[1]*height)
                w = int(detec[2]*width)
                h = int(detec[3]*height)
                #Bounding Box coord
                x = int(center_x -w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_Ids.append(class_Id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classNames[class_Ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 3)


    elapse_time = time.time() - timeNow
    fps = frame/elapse_time
    cv2.putText(img, "FPS " + str(fps), (10, 40), font, 1, (0,0,0), 1)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyWindow('Image')




