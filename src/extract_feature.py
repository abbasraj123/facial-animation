import csv
import cv2
import numpy as np


def write_csv(filename, data):
    with open(filename, 'w', newline="") as csvfile:
        writer = csv.writer(csvfile)
        for arow in data:
            writer.writerow(arow)


def get_fps(videofile):
    cap = cv2.VideoCapture(videofile)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (nFrame, fps)


def extract_one_frame_data(data, curPosition, nFrameSize, nSamPerFrame):
    frameData = np.zeros(nFrameSize, dtype=np.float32)
    if curPosition < 0:
        startPos = -curPosition
        frameData[startPos:nFrameSize] = data[0:(curPosition+nFrameSize)]
    else:
        chunk = data[curPosition:(curPosition+nFrameSize)]
        frameData[:len(chunk)] = chunk
    nextPos = curPosition + nSamPerFrame
    return (frameData, nextPos)
