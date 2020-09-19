from __future__ import print_function
import requests
import json
import cv2
import jsonpickle
from datetime import datetime
import argparse


# TODO: Refactor this code

server_address = 'http://localhost:8081'
test_url = server_address + '/api/predict'
color = (0,255,0)

parser = argparse.ArgumentParser()
opt = parser.parse_args()

VIDEO = 'video.mp4'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

# Get the divice
camera = cv2.VideoCapture(VIDEO)

# Used to analyze the speed of the camera
startTime = datetime.now()
timeprevious = datetime.now()
num_frames = 0

# Get the frames and analyze them
while True:

    # Get the current frame and the status
    status, frame = camera.read()

    # Stop condition
    key = cv2.waitKey(1) & 0xFF

    if (status == False or key == ord("q")):
        break

    img = frame.copy()

    _, img_encoded = cv2.imencode('.jpg', img)

    # Send image to analyze and getting tags as return
    response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
    # Get the score, labels and boxes from the response
    dictionary_obj = json.loads(response.content)['predictions']

    # If contains tags'
    if (dictionary_obj):
        predict_obj = dictionary_obj[0]
        # get the coordinates
        cX = int((predict_obj['boxes'][0] + predict_obj['boxes'][2]) / 2.0)
        cY = int((predict_obj['boxes'][1] + predict_obj['boxes'][3]) / 2.0)
        inputCentroids = (cX, cY)

        print(predict_obj['boxes'])

        # create bounding boxes
        cv2.rectangle(frame,(predict_obj['boxes'][0],predict_obj['boxes'][1]),(predict_obj['boxes'][2],predict_obj['boxes'][3]), color ,1)
        #put label on bounding box
        cv2.putText(frame,predict_obj['label'],(inputCentroids),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,cv2.LINE_AA)

    cv2.imshow("Frame", frame)

camera.release()
cv2.destroyAllWindows()