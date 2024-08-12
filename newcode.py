import os
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector

# Open the default camera
cap = cv2.VideoCapture(0)

# Set the frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = PoseDetector()

# Path to the folder containing shirt images
shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)
fixedRatio = 262 / 190  # widthOfShirt/widthOfPoint11to12
shirtRatioHeightWidth = 581 / 440
imageNumber = 0

# Load button images
imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10

def is_inside_rect(x, y, rect):
    return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]

def select_tshirt(event, x, y, flags, param):
    global imageNumber
    if event == cv2.EVENT_LBUTTONDOWN:
        x_offset = 50
        y_offset = 500
        for i in range(len(listShirts)):
            x_pos = x_offset + i * 110
            rect = (x_pos, y_offset, 100, 130)
            if is_inside_rect(x, y, rect):
                imageNumber = i
                break

def draw_catalog(img):
    x_offset = 50
    y_offset = 500
    for i, shirt in enumerate(listShirts):
        imgShirtThumb = cv2.imread(os.path.join(shirtFolderPath, shirt), cv2.IMREAD_UNCHANGED)
        imgShirtThumb = cv2.resize(imgShirtThumb, (100, 130))
        img = cvzone.overlayPNG(img, imgShirtThumb, (x_offset + i * 110, y_offset))
    return img

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

    if lmList:
        lm11 = lmList[11][1:3]
        lm12 = lmList[12][1:3]
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

        # Calculate the width of the shirt based on the distance between shoulders
        widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
        if widthOfShirt > 0:
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
            currentScale = (lm11[0] - lm12[0]) / 190
            offset = int(30 * currentScale), int(40 * currentScale)  # Adjusted offsets for better fit

            try:
                # Position the shirt based on the left shoulder point with adjusted offsets
                x = lm12[0] - offset[0]
                y = lm12[1] - offset[1]
                img = cvzone.overlayPNG(img, imgShirt, (x, y))
            except Exception as e:
                print(f"Error overlaying PNG: {e}")

    # Draw the t-shirt catalog
    img = draw_catalog(img)

    cv2.imshow("Image", img)
    
    # Check for mouse clicks to select a t-shirt
    cv2.setMouseCallback("Image", select_tshirt)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
