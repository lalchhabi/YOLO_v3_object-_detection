#### Importing libraries
import cv2
import numpy as np

#### Loading labels of objects
label_file = 'labels'
labels = []
with open(label_file,'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
print(len(labels))

#### Loading the configuration files and weights
#### yolov320 gives more accuracy but slow whereas yolov3-tiny is speed but gives less accuracy
modelconfig = 'yolov3-tiny.cfg'
modelweights = 'yolov3-tiny.weights'

##### creating network or model
net = cv2.dnn.readNetFromDarknet(modelconfig, modelweights)

#### Set backend as Opencv and processor as CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#### function for detecting object
confThreshold = 0.5
def findobjects(outputs, img):
    height, width, channels = img.shape
    bbox = []
    classid = []
    confs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence> confThreshold:
                w,h = int(detection[2] * width),  int(detection[3] * height) #### w = width , h= height
                x ,y = int((detection[0] * width) - w/2), int((detection[1] * height) - h/2) #### x,y = center point of bbox 
                bbox.append([x,y,w,h])
                classid.append(classId)
                confs.append(float(confidence))

    print(len(bbox))

    ##### Removing unwanted multiple bounding boxes using Non Max Suppression
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nms_threshold = 0.2)
    print(indices)
    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img, f'{labels[classid[i]].upper()} {int(confs[i]*100)}%', (x, y-10), 
                    cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255),2)
                       



### Loading webcam
cap = cv2.VideoCapture("cat.mp4")
width = 320
height = 320
while True:
    ret, img = cap.read()
    img = cv2.resize(img, (600,600))

    ##### converting image into blob so that model can load it.
    blob = cv2.dnn.blobFromImage(img, 1/255, (height, width),[0,0,0],1,crop = False)
    net.setInput(blob)

    #### Getting the layer Names
    layerNames = net.getLayerNames()
    # print(layerNames)
    #### Getting only output layer Name
    output_layers = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    # print(output_layers)
    ##### Getting the outputs of these three output layers
    outputs = net.forward(output_layers)
    # print(outputs[0].shape)  #### output of layer 82 => (no.of bounding box, no.of elements in y vector or output)
    # print(outputs[1].shape)  #### output of layer 94
    # print(outputs[2].shape) ##### output of layer 106
    # print(outputs[0][0])

    findobjects(outputs, img)






    cv2.imshow('Cat',img)

    k = cv2.waitKey(10)
    if k == ord('q') & 0xFF:
        break

cap.release()
cv2.destroyAllWindows()

