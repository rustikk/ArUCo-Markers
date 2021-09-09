import imutils
#VideoStream is used to access the webcam
from imutils.video import VideoStream
import argparse
import cv2
import time, sys

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL",
help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

#exception clause
if ARUCO_DICT.get(args["type"], None) is None:
    print(f"[INFO] ArUCo tag of {args['type']} is not supported")
    sys.exit(0)

#load the ArUCo dictionary and grab the ArUCo parameters
print(f"[INFO] detecting {args['type']} tags")
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

#start the video stream and allow the camera sensor to warm up
print(f"[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#loop over the frames from the video stream
while True:
    #grab the frame from the threaded video stream and resize it to have a
    #maximum width of 1000 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1000)

    #detect ArUCo markers in input frame
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    #exception clause to ensure program can work
    if len(corners) > 0:
        ids = ids.flatten()

        #loop over the detected ArUCo markers
        for (markerCorner, markerID) in zip(corners, ids):
            #extract the marker corners
            corners = markerCorner.reshape((4, 2))
            topLeft, topRight, bottomRight, bottomLeft = corners

            #convert x,y coordinates to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            #draw bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

            #calculate and draw the center x, y coordinates of the ArUCo marker
            cX = int((topLeft[0] + bottomLeft[0]) / 2.0)
            cY= int((topLeft[1] + bottomLeft[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

            #draw the ArUCo marker on the frame
            cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] -15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #break if 'q' is pressed
    if key == ord("q"):
        break

#clean up
cv2.destroyAllWindows()
vs.stop()

            