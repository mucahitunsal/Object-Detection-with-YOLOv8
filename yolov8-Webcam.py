from ultralytics import YOLO
import cv2
import cvzone
import random
# import math
import json
import os

with open('config.json') as config_file:
    config = json.load(config_file)

try:
    os.mkdir(config["paths"]["saved_Images_path"])
except OSError as error:
    print(error)

cap = cv2.VideoCapture(config["videoCapture"]["device"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["videoCapture"]["width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["videoCapture"]["height"])

model = YOLO(config["paths"]["model_Path"])

classNames = config["classes"]["classNames"]
# print(model)

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box

                x1, y1, x2, y2 = box.xyxy[0]
                w, h = int(x2) - int(x1), int(y2) - int(y1)
                bbox = int(x1), int(y1), int(w), int(h)
                # print(x1, y1, x2, y2)
                cvzone.cornerRect(frame, bbox, l=config["rectSetup"]["length"], t=config["rectSetup"]["thickness"],
                                  colorR=tuple(config["rectSetup"]["rectColor"]))

                # Confidence

                # confidence = math.ceil((box.conf[0] * 100)) / 100
                conff = round(float(box.conf[0]), 2)

                # Class name

                cls = box.cls[0]
                crClass = classNames[int(cls)]
                cvzone.putTextRect(frame, f'{crClass} {conff}', (max(0, int(x1)), max(35, int(y1))),
                                   scale=config["textSetup"]["scale"], thickness=config["textSetup"]["thickness"],
                                   offset=config["textSetup"]["offset"])

        key = cv2.waitKey(1)
        cv2.imshow("Video", frame)
        if key == ord("w"):
            cv2.imwrite(config["paths"]["saved_Images_path"] + "/saved_Image_{}.jpg".format(str(random.random())),
                        frame)
        elif key == ord("q"):
            break
        else:
            continue


else:
    print("Kamera Açılamıyor!")

cap.release()
cv2.destroyAllWindows()
