import cv2
import numpy as np

# Function to create coordinates for drawing a line
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]  # Bottom of the image
    y2 = int(y1 * (3 / 5))  # Slightly above the middle of the image
    x1 = int((y1 - intercept) / slope)  # Calculate x1 using the line equation
    x2 = int((y2 - intercept) / slope)  # Calculate x2 using the line equation
    return np.array([x1, y1, x2, y2])

# Function to average the slopes and intercepts of detected lane lines
def average_slope_intercept(image, lines):
    left_fit = []  # Store left lane line slopes and intercepts
    right_fit = []  # Store right lane line slopes and intercepts
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Fit a line (y = mx + c)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:  # If slope is negative, it's a left lane line
            left_fit.append((slope, intercept))
        else:  # If slope is positive, it's a right lane line
            right_fit.append((slope, intercept))
    
    # Take the average of left and right lane lines
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    # Convert averages to coordinates for drawing
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    
    return np.array([left_line, right_line])

# Function to apply Canny edge detection
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5,5), 0)  # Apply Gaussian blur to reduce noise
    canny = cv2.Canny(blur, 50, 150)  # Detect edges using Canny
    return canny

# Function to draw lines on an image
def display_lines(image, lines):
    line_image = np.zeros_like(image)  # Create a blank image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # Draw line in blue
    return line_image

# Function to define and apply a mask for selecting region of interest
def region_of_intrest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1000, height), (550, 250)]  # Define triangular region
    ])
    mask = np.zeros_like(image)  # Create a blank mask
    cv2.fillPoly(mask, polygons, 255)  # Fill the defined region with white (255)
    masked_image = cv2.bitwise_and(image, mask)  # Apply mask on the image
    return masked_image

# Process an image for lane detection
image = cv2.imread("C:/Users/jbsro/OneDrive/Desktop/Step1_Building_aSeflDrivingCar/test_image.jpg")
lane_image = np.copy(image)  # Make a copy of the image
canny_image = canny(lane_image)  # Apply Canny edge detection
cropped_image = region_of_intrest(canny_image)  # Apply region mask
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)  # Detect lines using Hough Transform
averaged_lines = average_slope_intercept(lane_image, lines)  # Average detected lines
line_image = display_lines(lane_image, averaged_lines)  # Draw lines
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)  # Overlay lane lines on original image
cv2.imshow("result", combo_image)  # Show final output
cv2.waitKey(0)  # Wait for user to close window

# Process a video for lane detection
cap = cv2.VideoCapture("C:/Users/jbsro/OneDrive/Desktop/Step1_Building_aSeflDrivingCar/test2.mp4")

while cap.isOpened():
    _, frame = cap.read()  # Read frame from video
    canny_image = canny(frame)  # Apply Canny edge detection
    cropped_image = region_of_intrest(canny_image)  # Apply region mask
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)  # Detect lines
    averaged_lines = average_slope_intercept(frame, lines)  # Average detected lines
    line_image = display_lines(frame, averaged_lines)  # Draw lines
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  # Overlay lane lines on original frame
    cv2.imshow("result", combo_image)  # Show final output
    
    # Stop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release video capture
cv2.destroyAllWindows()  # Close all OpenCV windows
