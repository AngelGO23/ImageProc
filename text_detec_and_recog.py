#This code was viewed from this youtube channel: https://www.youtube.com/channel/UCwVQ-caNvMMRh2fIlG48U1Q


from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract

#---------------------------------
'''This next two lines are for setting up the neural net that will be used to pass the
 photos that will be analyzed.'''
net = cv2.dnn.readNet('frozen_east_text_detection.pb')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
def text_detect_and_recognition(img):
    #This two lines are for preserving data (image, height, and width) of the original image
    org = img
    (H, W) = img.shape[:2]

    #Setting the image to the correct parameters to be turned later into a blob
    (newW, newH) = (640, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(img, (newW, newH))
    (H, W) = img.shape[:2]

    #Layers of the neural net for detecting where the text is in the image
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    #Modifiying the image to the correct format in order to pass it through the neural net
    blob = cv2.dnn.blobFromImage(img, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True,
                                 crop=False)
    net.setInput(blob)

    #These two variable represent the output of the neural net
    (scores, geometry) = net.forward(layerNames)
    (numRows, numCol) = scores.shape[2:4]
    rects = []
    confidences = []

    '''This  for loop is for choosing the rectangles that have sufficient
    confidence as to if it does has text inside the rectangle. We save the parameters
    of this rect.'''
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]

        xData0 = geometry[0, 0, y]

        xData1 = geometry[0, 1, y]

        xData2 = geometry[0, 2, y]

        xData3 = geometry[0, 3, y]

        anglesData = geometry[0, 4, y]

        for x in range(0, numCol):
            if scoresData[x] < 0.5:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY + (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    '''This for loop is for actually drawing the bounding box in the original image and
    passing that region of interest (roi) through the pytesseract functions.'''
    for (startX, startY, endX, endY) in boxes:
        print(startX, startY, endX, endY )
        newstartX = int(startX * rW)
        newstartY = int(startY * rH)
        newendX = int(endX * rW)
        newendY = int(endY * rH)
        boundary = 5

        roi = org[newstartY - boundary: newendY + boundary, newstartX - boundary: newendX + boundary]


        text = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        cong = r'--oem 2'
        textRecognized = pytesseract.image_to_string(text)

        textRecognized = textRecognized.replace("\n", "")
        textRecognized = textRecognized[:-1]
        cv2.rectangle(org, (startX, startY), (endX, endY), (0, 255, 0), 2)
        org = cv2.putText(org, textRecognized, (endX, endY+5), cv2.FONT_ITALIC, fontScale=0.5, color=(0, 0, 0))

    return org
'''Here is the main part that calls the functions for detecting and recognicing the text
in the image.'''
img0 = cv2.imread('Capture.PNG')
img1 = cv2.imread('streetSigns.jpg')
img2 = cv2.imread('doNotEnter.jpg')
img5 = cv2.imread('waySigns.jpeg')

array = [img0, img1, img2,img5]

for i in range(0, 2):
    for img in array:
        imgO = cv2.resize(img, (640, 320), interpolation=cv2.INTER_AREA)
        orig = cv2.resize(img, (640, 320), interpolation=cv2.INTER_AREA)
        textDetected = text_detect_and_recognition(imgO)
        cv2.imshow('Original Image', orig)
        cv2.imshow("Text Detected", textDetected)
        time.sleep(2)
        k = cv2.waitKey(0)
        if k == ord('q'):
            continue

cv2.destroyAllWindows()

