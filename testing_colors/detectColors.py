# Importing Libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

import requests
import cv2
import numpy as np
import imutils


def tryfunc():
      def nothing(x):
            pass

      # Load image
      image = cv2.imread('../pics/scr.png')

      # Create a window
      cv2.namedWindow('image')

      # Create trackbars for color change
      # Hue is from 0-179 for Opencv
      cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
      cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
      cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
      cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
      cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
      cv2.createTrackbar('VMax', 'image', 0, 255, nothing)

      # Set default value for Max HSV trackbars
      cv2.setTrackbarPos('HMax', 'image', 179)
      cv2.setTrackbarPos('SMax', 'image', 255)
      cv2.setTrackbarPos('VMax', 'image', 255)

      # Initialize HSV min/max values
      hMin = sMin = vMin = hMax = sMax = vMax = 0
      phMin = psMin = pvMin = phMax = psMax = pvMax = 0

      while (1):
            # Get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin', 'image')
            sMin = cv2.getTrackbarPos('SMin', 'image')
            vMin = cv2.getTrackbarPos('VMin', 'image')
            hMax = cv2.getTrackbarPos('HMax', 'image')
            sMax = cv2.getTrackbarPos('SMax', 'image')
            vMax = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and maximum HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(image, image, mask=mask)

            # Print if there is a change in HSV value
            if ((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (
                    pvMax != vMax)):
                  print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (
                        hMin, sMin, vMin, hMax, sMax, vMax))
                  phMin = hMin
                  psMin = sMin
                  pvMin = vMin
                  phMax = hMax
                  psMax = sMax
                  pvMax = vMax

            # Display result image
            cv2.imshow('image', result)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                  break

      cv2.destroyAllWindows()

class redObject:
  def __init__(self):
        # Capturing video
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # While loop to continuously fetching data from the Url
        while True:
              ret, frame = vid.read()
              frame = imutils.resize(frame, width=1200, height=2000)
              # Plotting four circles on the video of the object you want to        see the transformation of.


              # ColorSpace
              hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

              # Set range for red color
              # red_lower = np.array([160, 20, 70], np.uint8)
              # red_upper = np.array([69, 255, 255], np.uint8)
              # pink_lower = np.array([110, 50, 116], np.uint8)
              # pink_upper = np.array([197, 255, 255], np.uint8)
              red_lower = np.array([0, 190, 21], np.uint8)
              red_upper = np.array([78, 255, 255], np.uint8)
              red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
              if cv2.countNonZero(red_mask) > 0:
                    print('Red is present!')
              else:
                    print('No red!!!')
              red_mask = cv2.bitwise_and(frame, frame, mask=red_mask)
              cv2.imshow("red_mask", red_mask)
              # Set range for blue color
              blue_lower = np.array([45, 115, 103], np.uint8)
              blue_upper = np.array([123, 255, 255], np.uint8)

              blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
              blue_mask = cv2.bitwise_and(frame, frame, mask=blue_mask)
              cv2.imshow("blue_mask", blue_mask)


              if cv2.waitKey(1) & 0xff == 27:
                    cv2.destroyAllWindows()
