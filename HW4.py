from __future__ import print_function
import cv2
import numpy as np
import musicalbeeps
import math
import pyautogui
import argparse

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_blur = 'Blur kernel size'
window_name = 'HW4 window'
isColor = False

# part4
offsetP = -0.25
offsetA = -200
scaleP = 14
scaleA = 1/300

music_pitches = "A, B, C, D, E, F, G".split(",")

def nothing(x):
    pass


cam = cv2.VideoCapture(2)
print(music_pitches)
# cv2.namedWindow(window_name)
# cv2.createTrackbar(trackbar_type, window_name, 3, max_type, nothing)
# # Create Trackbar to choose Threshold value
# cv2.createTrackbar(trackbar_value, window_name, 0, max_value, nothing)
# # Call the function to initialize
# cv2.createTrackbar(trackbar_blur, window_name, 1, 20, nothing)
# # create switch for ON/OFF functionality
# color_switch = 'Color'
# cv2.createTrackbar(color_switch, window_name, 0, 1, nothing)
# cv2.createTrackbar('Contours', window_name, 0, 1, nothing)
while True:
    ret, frame = cam.read()

    # if not ret:
    #     break
    #
    # # 0: Binary
    # # 1: Binary Inverted
    # # 2: Threshold Truncated
    # # 3: Threshold to Zero
    # # 4: Threshold to Zero Inverted
    # threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    # threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    # blur_value = cv2.getTrackbarPos(trackbar_blur, window_name)
    # blur_value = blur_value + (blur_value % 2 == 0)
    # isColor = (cv2.getTrackbarPos(color_switch, window_name) == 1)
    # findContours = (cv2.getTrackbarPos('Contours', window_name) == 1)
    #
    # # convert to grayscale
    # if isColor == False:
    #     src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     _, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type)
    #     blur = cv2.GaussianBlur(dst, (blur_value, blur_value), 0)
    #     if findContours:
    #         _, contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)  # add this line
    #         output = cv2.drawContours(blur, contours, -1, (0, 255, 0), 1)
    #         print(str(len(contours)) + "\n")
    #     else:
    #         output = blur
    #
    #
    # else:
    #     _, dst = cv2.threshold(frame, threshold_value, max_binary_value, threshold_type)
    #     blur = cv2.GaussianBlur(dst, (blur_value, blur_value), 0)
    #     output = blur

    #### skinmask snippet
    lower_HSV = np.array([0, 40, 0], dtype="uint8")
    upper_HSV = np.array([25, 255, 255], dtype="uint8")

    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)

    lower_YCrCb = np.array((5, 140, 73), dtype="uint8")
    upper_YCrCb = np.array((255, 173, 133), dtype="uint8")

    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)

    skinMask = cv2.add(skinMaskHSV, skinMaskYCrCb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    # process the image to single out the hand
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # # use connected component analysis
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh, ltype=cv2.CV_16U)
    markers = np.array(markers, dtype=np.uint8)
    label_hue = np.uint8(179 * markers / np.max(markers))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    statsSortedByArea = stats[np.argsort(stats[:, 4])]
    if (ret > 2):
        try:
            roi = statsSortedByArea[-3][0:4]
            x, y, w, h = roi
            subImg = labeled_img[y:y + h, x:x + w]
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);
            _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            maxCntLength = 0
            for i in range(0, len(contours)):
                cntLength = len(contours[i])
                if (cntLength > maxCntLength):
                    cnt = contours[i]
                    maxCntLength = cntLength
            if (maxCntLength >= 5):
                ellipseParam = cv2.fitEllipse(cnt)
                # set cutoff to 40, which is the border of noise and detection
                cutoff = 50
                ###### filter out the situation where the x and y are too small(noise):
                # print(ellipseParam[0][0] + ellipseParam[0][1])
                ######## filter out situation where the MA vs ma is too extreme for a hand to form
                if ellipseParam[0][0] + ellipseParam[0][1] >= cutoff and ellipseParam[1][0] / ellipseParam[1][1] < 3 and ellipseParam[1][0] / ellipseParam[1][1] > 0.3:
                    # calculate the elements of the ellipse and incorporate into interface

                    # c_pitch is the ratio of MA and ma. The rounder the ellipse, the higher the pitch
                    c_pitch = ellipseParam[1][0] / ellipseParam[1][1]
                    if c_pitch > 0.8:
                        c_pitch = "G"
                    elif c_pitch > 0.65:
                        c_pitch = "A"
                    elif c_pitch > 0.5:
                        c_pitch = "B"
                    else:
                        c_pitch = "C5"

                    # c_note = music_pitches[c_pitch]

                    # c_amp is the distance of the hand to the camera, represented by the perimeter of the ellipse, calculated by MA and ma.
                    # the closer the hand is, the larger the ellipse perimeter, the higher the amp (louder the music note)
                    c_amp = scaleA * (offsetA + (2 * math.pi * math.sqrt((ellipseParam[1][0] ** 2 + ellipseParam[1][1] ** 2) / 2)))
                    if c_amp < 0:
                        c_amp = 0
                    elif c_amp > 1:
                        c_amp = 1

                    print("C_PITCH: ", c_pitch, "C_AMP: ", c_amp)
                    player = musicalbeeps.Player(volume=c_amp,
                                                 mute_output=False)
                    player.play_note(c_pitch, 0.2)

                # print("X: ", ellipseParam[0][0], "Y: ", ellipseParam[0][1], "MA: ", ellipseParam[1][0],
                #       "ma: ", ellipseParam[1][1], "angle: ", ellipseParam[2])
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB);
                subImg = cv2.ellipse(subImg, ellipseParam, (0, 255, 0), 2)

            subImg = cv2.resize(subImg, (0, 0), fx=3, fy=3)
            cv2.imshow("ROI " + str(2), subImg)
            k = cv2.waitKey(1)  # k is the key pressed
            if k == 27 or k == 113:  # 27, 113 are ascii for escape and q respectively
                # exit
                cv2.destroyAllWindows()
                cam.release()
                break
        except:
            a = 0

    # implement convex and contour
    # thresh = cv2.bitwise_not(thresh)
    # new_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    # _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # if len(contours) > 1:
    #     largestContour = contours[0]
    #     hull = cv2.convexHull(largestContour, returnPoints=False)
    #     fingerCount = 0
    #     for cnt in contours[:1]:
    #         defects = cv2.convexityDefects(cnt, hull)
    #         if (not isinstance(defects, type(None))):
    #             for i in range(defects.shape[0]):
    #                 s, e, f, d = defects[i, 0]
    #                 start = tuple(cnt[s][0])
    #                 end = tuple(cnt[e][0])
    #                 far = tuple(cnt[f][0])
    #
    #                 c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
    #                 a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2
    #                 b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2
    #                 angle = np.arccos((a_squared + b_squared - c_squared) / (2 * np.sqrt(a_squared * b_squared)))
    #
    #                 if angle <= np.pi / 3:
    #                     fingerCount += 1
    #                     cv2.circle(new_thresh, far, 4, [0, 0, 255], -1)
    #
    #                 cv2.line(new_thresh, start, end, [0, 255, 0], 2)
    #                 cv2.circle(new_thresh, far, 5, [0, 0, 255], -1)
    #     print("finger count: ", fingerCount)

    # cv2.imshow(window_name, thresh)
    #
    # k = cv2.waitKey(1)  # k is the key pressed
    # if k == 27 or k == 113:  # 27, 113 are ascii for escape and q respectively
    #     # exit
    #     cv2.destroyAllWindows()
    #     cam.release()
    #     break