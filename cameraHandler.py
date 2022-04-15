import math

import requests
import cv2
import numpy as np
import imutils


class cameraHandler:
  def __init__(self):

  # Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
    url = "http://192.168.68.103:8080/shot.jpg"

  # While loop to continuously fetching data from the Url
    while True:
      img_resp = requests.get(url)
      img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
      img = cv2.imdecode(img_arr, -1)
      img = imutils.resize(img, width=1000, height=1800)

      # Convert to grayscale
      cv2.imwrite("chess.png", img)
      # Read the original image
      img = cv2.imread("chess.png")
      # Display original image

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
      cv2.imshow('Canny Edge Detection', edges)

      cv2.imwrite("edges.png", edges)
      # Read the original image
      cv2.imread("edges.png")

      #Hough Lines

      default_file = "edges.png"
      #filename = argv[0] if len(argv) > 0 else default_file
      filename = "edges.png"
      # Loads an image
      src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)
      # Check if image is loaded fine
      if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
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

      #cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
      #cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
      # Display original image
      if cv2.waitKey(1) == 27:
        break
      cv2.imwrite("redLines.png", cdstP)
      # Read the original image
      image = cv2.imread("redLines.png")
      mask = np.zeros(image.shape, dtype=np.uint8)
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
      cnts = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      r = 0
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]
      valid_cnts = []
      v = []
      for c in cnts:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > 100 and area < 10000:
          x, y, w, h = cv2.boundingRect(c)
          cv2.drawContours(original, [c], -1, (36, 255, 12), 2)
          cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
          valid_cnts.insert(r,c)
          v.insert(r,[x,y,w,h])
          r = r + 1

      cv2.imshow("original", original)
      cv2.imshow("mask", mask)
      print("size {}".format( len(valid_cnts)))
      #print((sorted(v, key=lambda x:x[0], reverse=False)))
      i = 0
      if len(valid_cnts) == 64:
        for c in valid_cnts:
          if i < 9:
            cv2.drawContours(temp,[c] , -1, (255, 255, 255), -1)
          i+=1
      cv2.imshow("mask11", temp)
      cv2.imwrite("c1.png", temp)
      print(valid_cnts)
      #if len(valid_cnts) == 64:
        #break

  cv2.destroyAllWindows()

  #Code from: https://pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
  def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
      reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
      i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

