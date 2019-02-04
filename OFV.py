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
def overlay_flow(frame, flow, stepsize):
    if (frame.shape[0] <= stepsize or frame.shape[1] <= stepsize):
        print("Image height and width must be at least " + stepsize + " pixels")
        sys.exit()

    # want to quiver only subset of frame indices
    x_components = flow[..., 0]
    y_components = flow[..., 1]
    x_idxs = np.arange(stepsize, frame.shape[1], stepsize)
    y_idxs = np.arange(stepsize, frame.shape[0], stepsize)
    u = x_components[stepsize::stepsize, stepsize::stepsize]
    v = -y_components[stepsize::stepsize, stepsize::stepsize]
    plt.axis('equal')
    plt.quiver(x_idxs, y_idxs, u, v, scale=1, color='red', units='dots')
    plt.show(block=False)
    plt.pause(0.0001)
    plt.close()





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
    # frame1 = crop_frame(frame1)
    while (1):
        ret, frame2 = video.read()
        # frame2 = crop_frame(frame2)
        if (not ret):
            break
        flow_vector = get_optical_flow(frame1, frame2)
        plt.imshow(frame2)
        overlay_flow(frame1, flow_vector, 5)
        frame1 = frame2
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # np.set_printoptions(threshold=np.nan)
    video = cv2.VideoCapture('data/tennis_trimmed.mp4')
    play_video(video)
