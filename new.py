import cv2
import cvzone
import os
import numpy as np

# Initialize the camera and classifiers
cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Function to remove the white background from an image
def remove_white_background(img):
    if img.shape[2] == 4:  # If image already has an alpha channel (PNG)
        # Separate channels
        b, g, r, a = cv2.split(img)
        # Create a mask where the background is white
        mask = (b == 255) & (g == 255) & (r == 255)
        a[mask] = 0  # Set alpha to 0 for white background
        return cv2.merge([b, g, r, a])
    elif img.shape[2] == 3:  # If image does not have an alpha channel (JPEG)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        b, g, r = cv2.split(img)
        return cv2.merge([b, g, r, alpha])
    else:
        raise ValueError("Unsupported image format")

# Load all glass images into a list
glass_folder = 'Glasses'
glass_files = [f for f in os.listdir(glass_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

glasses = []
for f in glass_files:
    img = cv2.imread(os.path.join(glass_folder, f), cv2.IMREAD_UNCHANGED)
    img = remove_white_background(img)
    glasses.append(img)

selected_glass = 0
current_start_idx = 0

# Resize all sunglasses to a consistent size using high-quality interpolation
thumbnail_size = 100
glasses_resized = [cv2.resize(glass, (thumbnail_size, thumbnail_size), interpolation=cv2.INTER_AREA) for glass in glasses]

def draw_catalog(frame, glasses, start_idx, end_idx):
    catalog_height = 140
    spacing = 20
    x_offset = spacing + 50  # Extra space for left arrow

    for i in range(start_idx, end_idx):
        if i < len(glasses):
            thumbnail = glasses[i]
            y_offset = frame.shape[0] - catalog_height + 10

            # Ensure the thumbnail fits within the available space
            frame_height, frame_width = frame.shape[:2]
            available_height = min(thumbnail.shape[0], frame_height - y_offset)
            available_width = min(thumbnail.shape[1], frame_width - x_offset)

            if available_height > 0 and available_width > 0:
                frame[y_offset:y_offset+available_height, x_offset:x_offset+available_width] = thumbnail[:available_height, :available_width, :3]

            if i == selected_glass:
                cv2.rectangle(frame, (x_offset-5, y_offset-5), (x_offset+available_width+5, y_offset+available_height+5), (0, 255, 0), 2)

            x_offset += thumbnail_size + spacing

    # Draw arrows
    arrow_color = (200, 200, 200)
    cv2.putText(frame, '<', (20, frame.shape[0] - catalog_height + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 2, cv2.LINE_AA)
    cv2.putText(frame, '>', (x_offset + 10, frame.shape[0] - catalog_height + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, arrow_color, 2, cv2.LINE_AA)

def check_catalog_click(x, y, start_idx, end_idx, frame_height, frame_width):
    catalog_height = 140
    if y < frame_height - catalog_height:
        return None

    # Check for arrow clicks
    if 20 <= x <= 50 and frame_height - catalog_height <= y <= frame_height:
        return 'left_arrow'
    if frame_width - 50 <= x <= frame_width - 20 and frame_height - catalog_height <= y <= frame_height:
        return 'right_arrow'

    # Check for thumbnail clicks
    x_offset = 70
    for i in range(start_idx, end_idx):
        bx = x_offset
        by = frame_height - catalog_height + 10
        if bx + thumbnail_size > frame_width:
            break
        if bx <= x <= bx + thumbnail_size and by <= y <= by + thumbnail_size:
            return i
        x_offset += thumbnail_size + 20
    return None

def mouse_click(event, x, y, flags, param):
    global selected_glass, current_start_idx
    num_visible = 6  # Increase the number of visible images
    start_idx = current_start_idx
    end_idx = min(start_idx + num_visible, len(glasses))
    click_result = check_catalog_click(x, y, start_idx, end_idx, frame.shape[0], frame.shape[1])

    if event == cv2.EVENT_LBUTTONDOWN:
        if click_result == 'left_arrow':
            current_start_idx = max(0, current_start_idx - num_visible)
        elif click_result == 'right_arrow':
            if current_start_idx + num_visible < len(glasses):
                current_start_idx += num_visible
        elif isinstance(click_result, int):
            selected_glass = click_result

cv2.namedWindow('SnapLens', cv2.WINDOW_NORMAL)
cv2.resizeWindow('SnapLens', 1200, 800)  # Increase window size
cv2.setMouseCallback('SnapLens', mouse_click)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_scale)

    for (x, y, w, h) in faces:
        overlay = cv2.resize(glasses[selected_glass], (w, int(h*0.8)), interpolation=cv2.INTER_AREA)
        frame = cvzone.overlayPNG(frame, overlay, [x, y])

    num_visible = 4 # Display 6 images at a time
    start_idx = current_start_idx
    end_idx = min(start_idx + num_visible, len(glasses))

    draw_catalog(frame, glasses_resized, start_idx, end_idx)

    cv2.imshow('SnapLens', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
