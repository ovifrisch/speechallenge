import cv2
import skvideo.io
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import sys
# import matplotlib.pyplot as plt

# https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/

row_start = 40
row_end = 350

def crop_frame(frame):
    frame = frame[row_start:row_end, ...]
    return frame


# return an image with arrows indicating magnitiude and direction of flow component
# at every (stepsize) pixel in the frame


def overlay_arrow(frame, row, col, flow):
    x_comp = flow[0]
    y_comp = flow[1]
    return cv2.arrowedLine(frame, (col, row), (int(col + x_comp), int(row + y_comp)), [0, 0, 255])
    # return cv2.circle(frame, (col, row), 2, [0, 0, 255], -1)

# ss is stepsize
def overlay_flow(frame, flow, ss):
    for row in range(ss, flow.shape[0], ss):
        for col in range(ss, frame.shape[1], ss):
            frame = overlay_arrow(frame, row, col, flow[row, col, :])
    return frame


# adapted from https://github.com/JovanSardinha/speed-challenge-2017
def get_optical_flow(f1, f2):
    gray_f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray_f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # flow parameters
    flow_mat = None
    image_scale = 0.5
    levels = 1
    win_size = 15
    iterations = 2
    poly_n = 5
    STD = 1.1
    flags = 0

    return cv2.calcOpticalFlowFarneback(gray_f1, gray_f2, flow_mat, image_scale,
                                        levels, win_size, iterations,
                                        poly_n, STD, flags)

def play_video(video):
    _, frame1 = video.read()
    frame1 = crop_frame(frame1)
    while (1):
        ret, frame2 = video.read()
        frame2 = crop_frame(frame2)
        if (not ret):
            break
        flow_vector = get_optical_flow(frame1, frame2)
        overlayed_frame = overlay_flow(frame1, flow_vector, 10)
        # overlay = cv2.circle(frame1, (5, 5), 10, [0, 0, 255], -1)
        cv2.imshow('frame', overlayed_frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        frame1 = frame2
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # np.set_printoptions(threshold=np.nan)
    video = cv2.VideoCapture('data/trimmed.mp4')
    play_video(video)
