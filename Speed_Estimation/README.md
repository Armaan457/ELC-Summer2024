A speed estimation script that uses YOLOv8 to detect vehicles and DeepSORT for tracking purposes. Displays speed at a particular line to fix the perspective distortion issue. Presents a smile emoji when vehicle under speed limit and angry emoji when above speed limit. <br/>

To run this place the emoji's png in a folder named `assets` and the video in a folder name `data`. Based on the camera and it's placements, parameters like the speed line and ppm (pixels per meter) can be altered. Then finally run the `speed_estimate.py` file
