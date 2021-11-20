import cv2
import time
import numpy as np

# To Save The Output File In .avi (compatible for all os) Format
fourCC = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourCC, 20, (640, 480))

# Starting The Webcam
capture = cv2.VideoCapture(0)
# Allows Webcam to start by making the code sleep for 2 seconds
time.sleep(2)
bg = 0

# Capturing background for 60 frames
for i in range(60):
    ret,bg = capture.read()

# flipping the background
bg = np.flip(bg, axis=1)
# Reading The captured frame until the camera is open
while (capture.isOpened()):
    ret, img = capture.read()
    # if image is not captured then come out of the loop
    if not ret:
        break

    # Flipping The Image For Consistency
    img = np.flip(img, axis=1)
    # Converting the color value from BGR (RGB) to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Generating mask to detectred color

    # Adding some variation to the red color for mask 1
    lower_red = np.array([104, 153, 70])
    upper_red = np.array([30, 30, 0])
    # mask 1
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Adding some variation to the red color for mask 2
    lower_red = np.array([104, 153, 70])
    upper_red = np.array([30, 30, 0])
    # mask 2
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Adding Both Masks together
    mask1 = mask1 + mask2

    # Opening and expanding the Image where there is mask1 (red color)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Evry single bit not in red color is assigned to mask2
    mask2 = cv2.bitwise_not(mask1)

    # Keeping only the part of images without the red color
    res1 = cv2.bitwise_and(img, img, mask=mask2)
    # Keeping only the part of images with red color
    res2 = cv2.bitwise_and(bg, bg, mask=mask1)

    # Generating the Final Result by merging result1 and result 2
    final_res = cv2.addWeighted(res1, 1, res2, 1 ,0)
    output_file.write(final_res)

    # Displaying the output
    cv2.imshow("Invisibility Cloak", final_res)
    cv2.waitKey(1)

# Releasing the camera
capture.release()
# Destroying all windows
cv2.destroyAllWindows()