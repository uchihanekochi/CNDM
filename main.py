import os
import cv2 as cv


def get_frame_size(video_name):
    video_path = os.path.join('.', 'data','Sang VVK3.mp4')
    video_cap = cv.VideoCapture(video_path)
    width=video_cap.get(3)
    height=video_cap.get(4)
    print(width,height)

def draw_line(video_name):
    video_path = os.path.join('.', 'data','Sang VVK3.mp4')
    video_cap = cv.VideoCapture(video_path)
    ret,frame=video_cap.read()

    limits1 = [0, 370, 1080, 370]
    limits2 = [0, 600, 1080, 600]

    while ret:
        line1 = cv.line(frame, (limits1[0], limits1[1]), (limits1[2], limits1[3]), (0, 0, 255), 5)
        line2 = cv.line(frame, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 0, 255), 5)
        cv.imshow('frame',frame)
        cv.waitKey(1)
        ret,frame=video_cap.read()
    video_cap.release()
    cv.destroyAllWindows()



draw_line('VVK2.mp4')


