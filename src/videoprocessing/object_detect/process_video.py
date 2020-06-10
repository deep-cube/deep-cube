
from object_detect import *
import numpy as np 

input_video = cv2.VideoCapture('../../../data/video/kevin_single_moves_2.mp4')

# Check if camera opened successfully
if (input_video.isOpened()== False): 
  print("Error opening video stream or file")


# Set up the detector, can also use mode='segmentation'
detector = ObjectDetector(mode='segmentation', use_cpu=True)

detector.detect_video_and_save(input_video, frames=180, input_frame_rate=30, output_frame_rate=10, slow_factor=2, verbose=True, output_name='output.mp4')