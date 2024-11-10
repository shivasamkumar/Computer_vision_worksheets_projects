# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:43:55 2024

@author: shiva
"""

import cv2
from finger_count_functions import calc_accum_avg, segment, count_fingers

def main():
    accumulated_weight = 0.5
    roi_top, roi_bottom, roi_right, roi_left = 20, 300, 300, 600  # ROI values from the notebook
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        roi = frame[roi_top:roi_bottom, roi_right:roi_left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)  # GaussianBlur as per the notebook

        calc_accum_avg(gray, accumulated_weight)

        hand = segment(gray)
        if hand is not None:
            thresholded, hand_segment = hand
            fingers = count_fingers(thresholded, hand_segment)
            cv2.putText(frame, f"Fingers: {fingers}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.drawContours(frame, [hand_segment + (roi_right, roi_top)], -1, (0, 255, 0), 2)
            cv2.imshow("Thresholded", thresholded)

        cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 0, 255), 2)
        cv2.imshow("Finger Count", frame)

        
        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            break
          


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

