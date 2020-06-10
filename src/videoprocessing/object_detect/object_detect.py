##################################################
##     
##     
##     
##   NOTE: You need to install prerequisite libraries, or see environment.yml
##     
##   - python >= 3.6  
##   - pytorch >= 1.4
##   - pip install opencv-python
##   - pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
##   - python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
##   - pip install cython
##   - detectron2 (make sure your gcc/g++ ver > 5.0)
##     - if GPU: python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
##     - if CPU: python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html
##     
##     
##################################################



# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


class ObjectDetector():

    def __init__(self, mode='detection', use_cpu=True):
        """
        mode = "detection" / "segmentation"

        by default it use CPU, if use_cpu=False, it will use GPU (i.e. CUDA)
        """
        self.cfg = get_cfg()

        if use_cpu:
            self.cfg.MODEL.DEVICE='cpu'

        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        if mode == "detection":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        
        if mode == "segmentation":
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        self.predictor = DefaultPredictor(self.cfg)


    def detect_frame(self,im):
        """
        im: the image read from im = cv2.imread("img_path")

        @return the image array
        """
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = v.get_image()[:, :, ::-1]

        # img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # process img to RGB

        return img
        

    def detect_video_and_save(self, input_video, frames=180, input_frame_rate=30, output_frame_rate=10, slow_factor=2, verbose=True, output_name='output.mp4'):
        """
        input_video is the VideoCapture cv2 object
        """
        input_width  = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        input_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

        step_size = input_frame_rate // output_frame_rate
        curr_frame_i = 0

        out = cv2.VideoWriter(output_name,cv2.VideoWriter_fourcc('F','M','P','4'), output_frame_rate // slow_factor, (input_width,input_height))

        # Read until video is completed
        while(input_video.isOpened() and curr_frame_i < frames):
            
            # Capture frame-by-frame
            ret, frame = input_video.read()

            if ret == True:
                # detect object
                output_im = self.detect_frame(frame)
                output_frame = cv2.resize(output_im,(input_width,input_height))
                # write this frame
                out.write(output_frame)

                # cv2.imshow("Frame", output_im);
                #  # Press Q on keyboard to stop recording
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                if verbose:
                    print("Processed {}th frame...".format(curr_frame_i))

                # jump every step_size frame
                for _ in range(step_size-1):
                    ret, frame = input_video.read()
                    if ret == False: 
                        break
            else:
                break
            
            curr_frame_i += step_size


        out.release()
        cv2.destroyAllWindows() 

    def visualize(self,im):
        """
        im: either the image read from cv2.imread, or returned by detect

        press 'n' to close window
        press 's' to save the image to current directory and close window
        """
        cv2.imshow("curr_img", im)
        k = cv2.waitKey(0)
        if k == ord('n'):
            cv2.destroyAllWindows()
        elif k == ord('s'):
            cv2.imwrite('output.png',im)
            cv2.destroyAllWindows()
        elif k == ord('q'):
            cv2.destroyAllWindows()
            exit()














