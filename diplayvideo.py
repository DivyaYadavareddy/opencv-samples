import cv2
import imutils
from PIL import Image
from pytesseract import *
import numpy as np
# video_capture = cv2.VideoCapture(0)

# For real-time sample video detection
video_capture = cv2.VideoCapture("car.mp4")
video_width = video_capture.get(3)
video_height = video_capture.get(4)
fps = video_capture .get(cv2.CAP_PROP_FPS)
frame_count = int(video_capture .get(cv2.CAP_PROP_FRAME_COUNT))
print('frames per second : ' + str(fps))
print('Total number of frames : ' + str(frame_count))
while True:
    ret, frame = video_capture.read()
    plate_cascade = cv2.CascadeClassifier('./haarcascade_russian_plate_number.xml')
    frame = imutils.resize(frame, width=1000) # resize original video for better viewing performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert video to grayscale

    plate_rect = plate_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=7)

    for (x, y, w, h) in plate_rect:
        a, b = (int(0.02 * frame.shape[0]), int(0.025 * frame.shape[1]))  # parameter tuning
        plate = frame[y + a:y + h - a, x + b:x + w - b, :]
        # finally representing the detected contours by drawing rectangles around the edges.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 51, 255), 3)
        time = video_capture .get(cv2.CAP_PROP_POS_MSEC)
        time = time / 1000
        frame_no = video_capture .get(cv2.CAP_PROP_POS_FRAMES)
        if frame_no == 30 :
            print("number plate detected at frame:", plate_rect, "Frame number:", frame_no, "Time:", time)
            import cv2

            if __name__ == '__main__':
                # Read image
                # Select ROI
                r = cv2.selectROI(frame)

                # Crop image
                imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

                # Display cropped image
                cv2.imshow("Image", imCrop)
                cv2.waitKey(0)
                # im = Image.open('index.jpeg')"""
                imgCrop = cv2.resize(imCrop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpen = cv2.filter2D(imgCrop, -1, sharpen_kernel)
                text = image_to_string(imCrop)
                print("number plate is:")
                print("********")
                print(text)
                print("********")
                Dict = dict({"Detected at Frame No": frame_no, "Time": time})
                print(Dict)
    cv2.imshow('video',frame)
    # stop script when "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break