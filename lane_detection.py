import cv2
import numpy as np

#detects lane lines and midline by taking in frame of video
def detect_lines(frame):
    #initializes variables to later store avg coordinates of centerline
    x_coord = None
    y_coord = None

    #finds height and width from frame
    height, width = frame.shape[:2]

    #calculates coordinates of rectangle mask using frame height and width
    mask_size = max(height, width) // 2
    mask_left = int((width - mask_size) / 2)
    mask_top = int((height - mask_size) / 2)
    mask_right = mask_left + mask_size
    mask_bottom = mask_top + mask_size

    #pre processes frame with image processing using grayscale, blur, and canny edge
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.blur(gray_img, (3, 3))
    canny_img = cv2.Canny(blurred_img, 30, 100, apertureSize=3)

    #sets the mask to certain coordinates from canny_img
    mask = canny_img[mask_top:mask_bottom, mask_left:mask_right]

    #https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/
    #accessed 2/20
    #finds contours in the mask and uses only the most external pts
    contour_line, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fixed_contours = []

    #iterates through the contour coordinates to adjust them to place contour lines into mask
    for contour in contour_line:
        fix_contour = contour.copy()
        fix_contour[:, 0, 0] += mask_left
        fix_contour[:, 0, 1] += mask_top
        fixed_contours.append(fix_contour)

    #draws contour lines onto frame
    contoured_img = cv2.polylines(frame, fixed_contours, isClosed=False, color=(0, 0, 255), thickness=5)

    counter = 0

    #https://stackoverflow.com/questions/64396183/opencv-find-a-middle-line-of-a-contour-python
    #accessed 2/24
    #checks if at least two contour lines detected in the mask
    if len(contour_line) >= 2:

        centerline = []

        #finds min length of first and last contours
        first_contour_len = len(contour_line[0])
        last_contour_len = len(contour_line[-1])
        min_contour = min(first_contour_len, last_contour_len)

        #iterates over min contour length
        for i in range(min_contour):
            #calculates avg x and y coords for each pt of midline by taking avg of pts from first and last contour
            x_coord = (contour_line[0][i][0][0]+contour_line[-1][i][0][0]) // 2 + mask_left
            y_coord = (contour_line[0][i][0][1]+contour_line[-1][i][0][1]) // 2 + mask_top
            centerline.append((x_coord, y_coord))

        #puts centerline coordinates into an array
        centerline_coord = np.array([centerline])

        #draws centerline
        contoured_img = cv2.polylines(contoured_img, [centerline_coord], isClosed=False, color=(0, 0, 255), thickness=8)
    else:
        #if less than 2 contour lines detected, checks last available midline coords and draws a midline
        if counter < 11 and x_coord is not None and y_coord is not None:
            cv2.polylines(contoured_img, [(x_coord, y_coord)], isClosed=False, color=(0, 0, 255), thickness=8)
            counter += 1

    #draws rectangle around rectangle mask
    cv2.rectangle(frame, (mask_left, mask_top), (mask_right, mask_bottom), (255, 255, 255), 2)

    #returns frame with lines and rectangle mask drawn
    return frame
