import colorsys
from checkers.constants import *
from checkers.board import *
import math
import requests
import cv2
import numpy as np
import imutils
from imutils import contours
from scipy.spatial import distance as dist
from checkers.piece import Piece

# we use the matrix to detect illegal moves in the game flow
legalCells = np.array([
                  0, 1, 0, 1, 0, 1, 0, 1
                 ,1, 0, 1, 0, 1, 0, 1, 0
                 ,0, 1, 0, 1, 0, 1, 0, 1
                 ,1, 0, 1, 0, 1, 0, 1, 0
                 ,0, 1, 0, 1, 0, 1, 0, 1
                 ,1, 0, 1, 0, 1, 0, 1, 0
                 ,0, 1, 0, 1, 0, 1, 0, 1
                 ,1, 0, 1, 0, 1, 0, 1, 0])

# this class is where all the function of the board detection are defined
class Board_Detector:
    def __init__(self):
        self.originalImg = None
        self.boardAngles = None
        self.boardContours = None
        self.mask = None
        self.maskcopy = None
        self.checkerboard_r = None

    # this is the first in the detection process
    # we use it in order to detect the board cells postions
    def findBoard(self):
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, img = vid.read()
            img = imutils.resize(img, width=1200, height=2000)
            # Convert to grayscale
            cv2.imwrite("pics/checkers_board.png", img)
            # Read the original image
            img = cv2.imread("pics/checkers_board.png")
            self.originalImg = img
            # Display original image
            temp1 = img.copy()
            # Convert to graycsale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Blur the image for better edge detection
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

            # Sobel Edge Detection
            cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
            cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
            cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
            # Display Sobel Edge Detection Images

            # Canny Edge Detection
            edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
            # Display Canny Edge Detection Image
            #cv2.imshow('Canny Edge Detection', edges)

            cv2.imwrite("pics/edges.png", edges)
            # Read the original image
            cv2.imread("pics/edges.png")

            # Hough Lines

            default_file = "pics/edges.png"
            # filename = argv[0] if len(argv) > 0 else default_file
            filename = "pics/edges.png"
            # Loads an image
            src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)
            # Check if image is loaded fine
            if src is None:
                #print('Error opening image!')
                #print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
                return -1

            dst = cv2.Canny(src, 50, 200, None, 3)

            # Copy edges to the images that will display the results in BGR
            cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
            cdstP = np.copy(cdst)

            lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

            if lines is not None:
                for i in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                    cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

            linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

            if linesP is not None:
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imwrite("pics/redLines.png", cdstP)
            # Read the original image
            image = cv2.imread("pics/redLines.png")
            mask = np.zeros(image.shape, dtype=np.uint8)
            self.maskcopy = mask.copy()
            temp = mask.copy()
            original = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # Remove noise with morph operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            invert = 255 - opening

            # Find contours and find squares with contour area filtering + shape approximation
            cnts = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            # Take each row of 8 and sort from left-to-right
            (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
            r = 0


            #sort_contours(cnts, "left-to-right")
            #sort_contours(cnts, "bottom-to-top")
            valid_cnts = []
            v = []
            areas = []
            for c in cnts:
                area = cv2.contourArea(c)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4 and area > 750 and area < 30000:
                    areas.append(area)
                    cv2.drawContours(original, [c], -1, (36, 255, 12), 2)
                    cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
                    valid_cnts.insert(r, c)
                    r = r + 1


            checkerboard_row = []
            row = []
            for (i, c) in enumerate(valid_cnts, 1):
                row.append(c)
                if i % 8 == 0:
                    (valid_cnts, _) = contours.sort_contours(row, method="left-to-right")
                    checkerboard_row.append(valid_cnts)
                    row = []
            r = 0
            for row in checkerboard_row:
                for c in row:
                    x, y, w, h = cv2.boundingRect(c)
                    s = img[y:y + h, x:x + w]
                    imgStr = "squares/square" + str(r) + ".png"
                    v.insert(r, [x, y, w, h])  # angles array
                    cv2.imwrite(imgStr, s)
                    r=r+1

            cv2.imshow('maskSorted', self.maskcopy)
            cv2.imshow("mask", mask)
            #print("size {}".format(len(valid_cnts)))
            i = 0
            myLen = 0
            for e in checkerboard_row:
                myLen+=len(e)
            if myLen == 64:
                self.boardAngles = v
                self.boardContours = valid_cnts
                self.checkerboard_r = checkerboard_row
                self.mask = mask
                # for c in valid_cnts:
                #     if i < 10:
                #         cv2.drawContours(temp, [c], -1, (255, 255, 255), -1)
                #     i = i + 1

                #cv2.imwrite("firstTen.png", temp)
                cv2.imwrite("pics/mask.png", mask)
                #self.drawSquares()
                break

            if cv2.waitKey(1) == 27:
                break


        cv2.destroyAllWindows()

    # this function is used to update the squares images to the current state
    # integer matrix that tells us which color is in each cell
    def updateSquareImgs(self):
      r = 0
      vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
      ret, img = vid.read()
      img = imutils.resize(img, width=1200, height=2000)
      self.originalImg = img
      mat = []
      rowList=[]
      for angles in self.boardAngles:
          x, y, w, h = angles
          s = self.originalImg[y:y + h, x:x + w]
          imgStr = "squares/square" + str(r) + ".png"
          cv2.imwrite(imgStr, s)
          color = self.readImageColor(r)
          rowList.append(color)
          if len(rowList) == 8:
              mat.append(rowList)
              rowList = []
          r = r + 1
      return mat

    # Checks the image's color and updates the squares array in palce = {id}.
    # If square is empty it updates it to zero
    #         0 == empty
    #         1 == pink
    #         2 == blue
    #         -2 == illegal pink
    #         -1 == illegal blue
    def readImageColor(self, id):
        imgStr = "squares/square" + str(id) + ".png"
        frame = cv2.imread(imgStr)
        frame = imutils.resize(frame, width=1200, height=2000)
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # blue_lower = np.array([98, 91, 116], np.uint8)
        blue_lower = np.array([85, 45, 47], np.uint8)
        blue_upper = np.array([160, 255, 255], np.uint8)
        # blue_lower = np.array([0, 124, 0], np.uint8)
        # blue_upper = np.array([123, 255, 255], np.uint8)
        # blue_lower = np.array([98, 91, 116], np.uint8)
        # blue_upper = np.array([165, 255, 255], np.uint8)
        # blue_upper = np.array([130, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
        if cv2.countNonZero(blue_mask) > 0:
            if not legalCells[id]:
                #print("Illegal move in square{}".format(id))
                return -1
            #print('The image idx = {} is blue!'.format(id))
            blue_mask = cv2.bitwise_and(frame, frame, mask=blue_mask)
            # cv2.imshow("blue_mask", blue_mask)
            return 2
        else:
            # Set range for red color
            # red_lower = np.array([160, 20, 70], np.uint8)
            # red_upper = np.array([69, 255, 255], np.uint8)
            # pink_lower = np.array([0, 190, 21], np.uint8)
            # pink_upper = np.array([78, 255, 255], np.uint8)
            pink_lower = np.array([110, 50, 116], np.uint8)
            pink_upper = np.array([197, 255, 255], np.uint8)
            # (36, 25, 25), (70, 255,255)
            pink_mask = cv2.inRange(hsvFrame, pink_lower, pink_upper)
            if cv2.countNonZero(pink_mask) > 0:
                if not legalCells[id]:
                    #print("Illegal move in square{}, area = {}".format(id, cv2.countNonZero(pink_mask)))
                    return -2
                #print('The image idx = {} is yellow!'.format(id))
                pink_mask = cv2.bitwise_and(frame, frame, mask=pink_mask)
                # cv2.imshow("yellow_mask", yellow_mask)
                return 1
            else:
                #print("Square idx = {} is empty".format(id))
                return 0

    # this function calls the updateSquareImage and it updates the sorted.png image to the current state
    # integer matrix which tells us which color is in each square
    # we use this function in order to get the new state of the board after each move and update the images
    def drawSquares(self):
        # Draw text
        mat = self.updateSquareImgs()
        number = 0
        tempImg = self.maskcopy.copy()
        for row in self.checkerboard_r:
            for c in row:

                M = cv2.moments(c)
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
                blue = (255, 0, 0)
                yellow = (255, 0, 255)
                white = (255, 255, 255)
                red = (0, 0, 255)
                """      
                         0 = empty
                         1 = yellow
                         2 = blue
                         -1 = illegal
                """
                res = self.readImageColor(number)
                if res == 0:
                    cv2.putText(tempImg, "w {}".format(number + 1), (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                white, 2)
                if res == 1:
                    cv2.putText(tempImg, "p {}".format(number + 1), (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                yellow, 2)
                if res == 2:
                    cv2.putText(tempImg, "b {}".format(number + 1), (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                blue, 2)
                if res == -1:
                    cv2.putText(tempImg, "{}".format("  X "), (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                red, 2)
                if res == -2:
                    cv2.putText(tempImg, "{}".format("  X "), (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                red, 2)
                number += 1
        cv2.imwrite("pics/sorted.png", tempImg)
        cv2.imshow('sortedSquares', tempImg)
        return mat
