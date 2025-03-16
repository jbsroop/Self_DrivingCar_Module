# Lane Detection using OpenCV

## Overview
This project implements a simple lane detection system using OpenCV and NumPy. It processes both images and videos to detect lane lines on roads using Canny edge detection, Hough Transform, and line averaging techniques.

## Features
- Detects lane lines in an image or a video.
- Uses edge detection and region of interest masking.
- Applies Hough Line Transform for line detection.
- Averages multiple detected lines for a smoother output.

## Requirements
Ensure you have the following dependencies installed:
```sh
pip install opencv-python numpy
```

## Usage

### For Image Processing
1. Place your test image in the desired directory.
2. Update the image path in the code:
   ```python
   image = cv2.imread("path/to/your/image.jpg")
   ```
3. Run the script, and the detected lane lines will be displayed.

### For Video Processing
1. Place your test video in the desired directory.
2. Update the video path in the code:
   ```python
   cap = cv2.VideoCapture("path/to/your/video.mp4")
   ```
3. Run the script, and lane detection will be applied to each frame of the video.
4. Press 'q' to exit the video processing loop.

## How It Works
1. **Canny Edge Detection**: Detects edges in the image.
2. **Region of Interest Masking**: Focuses on the relevant area (road lanes).
3. **Hough Line Transform**: Identifies lines from the edge-detected image.
4. **Averaging and Drawing Lines**: Computes an average line for both left and right lanes and overlays them on the image/video.

## Output
- The final result is displayed as an image or video with lane lines highlighted in NavyBlue.
- The output image has weighted blending to make the lines appear naturally overlaid on the road.

## Notes
- Ensure the correct paths are set for the image and video files.
- The region of interest is defined based on assumptions; adjust it if necessary for different road conditions.
- Parameters like Canny threshold values and Hough Transform parameters can be fine-tuned for better accuracy.

## Example
Run the script and check the displayed output:
```sh
python lane_detection.py
```
The result will be shown in an OpenCV window.


