'''This code is originally from this youtube video: https://www.youtube.com/watch?v=WQeoO7MI0Bs
I have added personal comments on the functions of each part
Feel free to correct any errors;)'''
#-----------------------
#Only two libraries needed
import cv2
import numpy as np

#-----------------------
'''Upoading the video from the camera specified
by the function cv2.VideoCapture(0). The number 0 is what identifies what camera to use (or what 
video file to use ;).'''
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
#-----------------------
'''This are the colors that the program will Identify.
Each element is a list of 6 elements in which the first 3 are
the lower limits fot Hue, saturation, and value, respectively. 
The last 3 are the upper limits, respectively'''
myColors = [
    [5, 107, 0, 19, 255, 255],
    [133, 56, 0, 159, 156, 255],
    [57, 76, 0, 100, 255, 255]
]
#-----------------------
''''''
colorVal = [
    [51, 153, 255],
    [255, 0, 255],
    [0, 255, 0]
]

myPoints = []

#-----------------------
'''This function finds the colors using the lower and upper limits
in the variable colorValue which in this case we pass the variable myColors'''
def findColor(img, myColor, colorValue):
    #Convert the image into HSV to find the colors
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #The count is used for having the correct coloring point,
    # using count as the index in the colorVal list
    count = 0
    #This is the list of points that will be returned in order to draw them on the window
    newPoints = []
    for color in myColor:

        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        cv2.circle(imgResult, (x, y), 10, colorValue[count], cv2.FILLED)
        if x != 0 and y != 0:
            newPoints.append([x, y, count])
        count +=1

    return newPoints
#--------------------------
'''This functions finds the "Bounding box" for the object with the
one of the colors specified in myColors. It returns the point where the "tip" 
that the object will draw off from the colored object will be.
'''
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            #cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y
#-------------------------
'''This last function is where we draw the points using the list of the global variable myPoints'''
def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)
#-------------------------
'''This where put all the code together. The while loop is for displaying continually 
the frames(one IMAGE at a time) of the video'''
while True:
    succes, img = cap.read()
    imgResult = img.copy()
    newPoints = findColor(img, myColors, colorVal)
    '''We check if there is a need for drawing points with these
    two if statements'''
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, colorVal)
    cv2.imshow("Result", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
