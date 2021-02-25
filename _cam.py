import cv2, threading, time, os, datetime
import numpy as np

"""
    NEURAL NETWORK FRAMEWORK: AlexeyAB'S YOLO DARKNET
    MODEL TRAINED IN: GOOGLE COLAB (FREE GPU)
    OBJECT DETECTION ALGORITHM: YOLO (YOU ONLY LOOK ONCE)
    LIBRARIES USED:
        - cv2
        - threading
        - time
        - os
        - dateime
        - numpy
"""

class ObjectDetection (object):

    def __init__(self, weights, cfg, names):
        self.net = cv2.dnn.readNet(weights, cfg)
        self.classes = []
        with open(names, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        self.outputLayers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.currentDetected = ()
        self.image = None

        self.detecting = True

        camera = threading.Thread(target=self.videoCamera)
        detection = threading.Thread(target=self.detectObjects)
        
        camera.start()
        detection.start()
        camera.join()
        detection.join()
        
    def detectObjects(self):
        while self.detecting:
            
            if self.image is not None:
                height, width, channels = self.image.shape
                blob = cv2.dnn.blobFromImage(self.image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                self.net.setInput(blob)
                outs = self.net.forward(self.outputLayers)
                class_ids = []
                confidences = []
                square = []
                
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        del scores
                        if confidence > 0.5:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w/2)
                            y = int(center_y - h/2)
                            square.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                            del confidence, class_id, center_x, center_y, w, h, x, y

                del outs
                indexes = cv2.dnn.NMSBoxes(square, confidences, 0.5, 0.5)
                self.currentDetected = (square, class_ids, confidences, indexes)

    def videoCamera(self):
        capture = cv2.VideoCapture(0)
            
        font = cv2.FONT_HERSHEY_PLAIN
        
        while True:
            
            ret, self.image = capture.read()
            cv2.imshow("image", self.image)
            
            if len(self.currentDetected) > 0:
                square = self.currentDetected[0]
                class_ids = self.currentDetected[1]
                confidences = self.currentDetected[2]
                indexes = self.currentDetected[3]
                
                for i in range(len(square)):
                    
                    if i in indexes:
                        x, y, w, h = square[i]
                        label = self.classes[class_ids[i]]
                        confidence = round(confidences[i], 2)
                        cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(self.image, "{} {}".format(label, confidence), (x, y), font, 1, (0, 255, 0), 1)
                        
            cv2.imshow("image", self.image)
            key = cv2.waitKey(1)
            
            if key == 27:
                break

        capture.release()
        cv2.destroyAllWindows()
        self.detecting = False

if __name__=="__main__":
    
    if not os.path.exists("config.txt"):
        print("> config.txt is missing!")
        os.system("pause")
        exit()

    with open("config.txt", "r") as file:
        args = [f.strip() for f in file.readlines()]

    if len(args) < 3:
        print("> cfg.txt is either empty or has missing argument\\s!\n> It needs 3 argument files (in order):\n> .weights\n> .names\n> .cfg")
        os.system("pause")
        exit()

    yoloDetection = ObjectDetection(args[0], args[1], args[2])
