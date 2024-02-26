# there is cited code in the lane_detection.py file!
import cv2
from lane_detection import detect_lines

#takes in video feed from camera
video_feed = cv2.VideoCapture(0)

#continuosly capture frames from video_feed
while True:

    #reads frames
    ret, frame = video_feed.read()

    #calls detect_lines function and applies to frame
    updated_frame_with_lines = detect_lines(frame)

    #resizes frame to fit window
    display_frame = cv2.resize(updated_frame_with_lines, dsize=(700, 400))

    #creates window to display lines and mask on frame
    cv2.imshow('lane detection', display_frame)

    #ends program if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#releases video_feed and closes opencv windows
video_feed.release()
cv2.destroyAllWindows()
