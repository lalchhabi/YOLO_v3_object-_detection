#### Importing all trhe 
import cv2
import numpy as np
import matplotlib.pyplot as plt

#### Loading the weights and configuration files
weights = r'yolov3_custom_2000.weights'
cfg = r'yolov3_custom.cfg'
net = cv2.dnn.readNetFromDarknet(cfg, weights)

#### Loading classes or labels
classes = []
with open('classes.names','rt') as f:
    classes = f.read().splitlines()

#### Reading an image
img = cv2.imread('test/000350_jpg.rf.6beb7c1c5b00a02a97f019079799dc57.jpg')
img = cv2.resize(img, (700,700))
height, width, channels = img.shape

#### Convert image into blob and load it on model
blob =cv2.dnn.blobFromImage(img, 1/255, (height, width), (0,0,0), swapRB = True, crop =False)
net.setInput(blob)

#### Getting all the three detection layers of yolo
output_layers_names = net.getUnconnectedOutLayersNames()
# print(output_layers_names)
layersOutputs = net.forward(output_layers_names)
# print(layersOutputs)

#### Finding the y-vector and minimum no.of bounding box
confthreshold = 0.5
boxes = []
confidences = []
class_ids = []

for output in layersOutputs:

    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confthreshold:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

##### Applying Non max Suppression for removing unwanted multiple bounding boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, confthreshold, nms_threshold = 0.3)
for i in indexes:
    box = boxes[i]
    x,y,w,h = box
    cv2.rectangle(img, (x,y),(x+w, y+h), (0,255,0),2)
    conf_value = str(round(confidences[i],2))
    label = str(classes[class_ids[i]])
    cv2.putText(img, label + " " + conf_value, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 2,(0,255,0),2)

cv2.imshow('Final Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()