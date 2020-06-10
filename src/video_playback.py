import cv2
import os
import json

# a quick hack to examine a video frame by frame to manually
# label a sequence of data for sanity check

FILEPATH = '../data/video/sanity3.mp4'


def play_video():
    # load video capture from file
    video = cv2.VideoCapture(FILEPATH)
    # window name and size
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

    frame_num = 0
    while video.isOpened():
        print(frame_num)

        # Read video capture
        ret, frame = video.read()
        # Display each frame
        cv2.imshow("video", frame)
        # show one frame at a time
        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('k'), ord(' ')]:
            key = cv2.waitKey(0)
        # Quit when 'q' is pressed
        if key == ord('q'):
            break

        frame_num += 1

    # Release capture object
    video.release()
    # Exit and distroy all windows
    cv2.destroyAllWindows()

    print(frame_num)


if __name__ == "__main__":
    play_video()
