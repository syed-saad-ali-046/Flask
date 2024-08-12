import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Load a watch image with a transparent background
watch_img_path = "watch.png"

if not os.path.isfile(watch_img_path):
    raise FileNotFoundError(f"{watch_img_path} not found. Please check the file path.")

watch_img = cv2.imread(watch_img_path, cv2.IMREAD_UNCHANGED)

# Check if the image was loaded successfully
if watch_img is None:
    raise RuntimeError("Failed to load the watch image. Ensure the file is in the correct location and readable.")

# Print image shape for debugging
print(f"Watch image loaded with shape: {watch_img.shape}")

# Ensure the image has an alpha channel
if watch_img.shape[2] == 3:
    watch_img = cv2.cvtColor(watch_img, cv2.COLOR_BGR2BGRA)

# Function to overlay the watch image
def overlay_image(background, overlay, x, y, scale=1, angle=0):
    overlay = cv2.resize(overlay, None, fx=scale, fy=scale)
    
    # Rotate the overlay image
    (h, w) = overlay.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    overlay = cv2.warpAffine(overlay, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    h, w, _ = overlay.shape

    # Ensure the overlay image fits within the background boundaries
    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]
    
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h]

    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] +
                                       alpha_background * background[y:y+h, x:x+w, c])
    return background

# Function to calculate the angle between two points
def calculate_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    angle = np.arctan2(delta_y, delta_x)
    return np.degrees(angle)

# Main function to capture video and process frames
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video capture device.")
        
    while True:
        success, img = cap.read()
        if not success:
            break

        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image and detect hands
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get wrist coordinates
                wrist_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * img.shape[1])
                wrist_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * img.shape[0])
                
                # Get other points to calculate angle (e.g., index finger base and pinky base)
                index_finger_base_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * img.shape[1])
                index_finger_base_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * img.shape[0])
                pinky_base_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * img.shape[1])
                pinky_base_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * img.shape[0])
                
                # Calculate the angle of the wrist
                angle = calculate_angle((pinky_base_x, pinky_base_y), (index_finger_base_x, index_finger_base_y))
                
                # Calculate the distance for scaling
                distance = np.sqrt((index_finger_base_x - wrist_x)**2 + (index_finger_base_y - wrist_y)**2)
                
                # Scale based on the distance
                scale_factor = distance / 150.0  # Adjust 150.0 as needed based on your image size
                
                # Positioning and rotating the watch
                img = overlay_image(img, watch_img, wrist_x - int(watch_img.shape[1] * scale_factor) // 2,
                                    wrist_y - int(watch_img.shape[0] * scale_factor) // 2, 
                                    scale=scale_factor, angle=angle)

        # Display the resulting frame
        cv2.imshow('Wrist Watch Detection', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
