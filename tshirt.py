import cv2
import numpy as np
import imutils
from imutils import perspective
from scipy.spatial import distance

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def calculate_size_ratios(measurements):
    standard_ratios = {
        'S': {'chest_to_waist': 1.1, 'back_to_sleeve': 0.95, 'front_to_back': 1.05},
        'M': {'chest_to_waist': 1.15, 'back_to_sleeve': 1.0, 'front_to_back': 1.1},
        'L': {'chest_to_waist': 1.2, 'back_to_sleeve': 1.05, 'front_to_back': 1.15},
        'XL': {'chest_to_waist': 1.25, 'back_to_sleeve': 1.1, 'front_to_back': 1.2},
        'XXL': {'chest_to_waist': 1.3, 'back_to_sleeve': 1.15, 'front_to_back': 1.25}
    }
    
    chest_to_waist_ratio = measurements['chest'] / measurements['waist']
    back_to_sleeve_ratio = measurements['back_length'] / measurements['sleeve']
    front_to_back_ratio = measurements['front_length'] / measurements['back_length']

    size_scores = {}
    for size, ratios in standard_ratios.items():
        size_scores[size] = (
            abs(chest_to_waist_ratio - ratios['chest_to_waist']) +
            abs(back_to_sleeve_ratio - ratios['back_to_sleeve']) +
            abs(front_to_back_ratio - ratios['front_to_back'])
        )
    
    best_size = min(size_scores, key=size_scores.get)
    
    return best_size

def get_measurements(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Image not found!'}
        
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if len(cnts) == 0:
            return {'error': 'No contours found!'}
        
        c = max(cnts, key=cv2.contourArea)
        ((x, y), (w, h), angle) = cv2.minAreaRect(c)
        
        box = cv2.boxPoints(((x, y), (w, h), angle))
        box = np.array(box, dtype="int")
        
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        
        tltrX, tltrY = midpoint(tl, tr)
        blbrX, blbrY = midpoint(bl, br)
        tlblX, tlblY = midpoint(tl, bl)
        trbrX, trbrY = midpoint(tr, br)
        
        chest = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
        waist = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))
        back_length = distance.euclidean((tl[0], tl[1]), (bl[0], bl[1]))
        sleeve = distance.euclidean((tr[0], tr[1]), (br[0], br[1]))
        
        # New front length calculation: Top to bottom along the front side
        front_length = distance.euclidean((tr[0], tr[1]), (br[0], br[1]))

        pixelsPerMetric = max(chest, waist, back_length, sleeve, front_length) / 24.0
        
        measurements = {
            'chest': chest / pixelsPerMetric,
            'waist': waist / pixelsPerMetric,
            'back_length': back_length / pixelsPerMetric,
            'sleeve': sleeve / pixelsPerMetric,
            'front_length': front_length / pixelsPerMetric  
        }
        
        size = calculate_size_ratios(measurements)
        measurements['size'] = size
        
        return measurements
    except Exception as e:
        return {'error': str(e)}

def process_upload(filepath):
    return get_measurements(filepath)
