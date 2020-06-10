
from object_detect import *
import numpy as np 


cap = cv2.VideoCapture('../../../data/video/kevin_single_moves_2.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Set up the detector, can also use mode='segmentation'
detector = ObjectDetector(mode='segmentation')

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    detector.visualize(frame)
    # cv2.imshow('Frame',frame)    
    output_im = detector.detect_frame(frame)
    detector.visualize(output_im)

    # Press Q on keyboard to exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #   break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()