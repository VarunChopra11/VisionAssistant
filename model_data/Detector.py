import cv2
import numpy as np
import requests
import threading
import time
import pygame
from queue import Queue

np.random.seed(20)

KNOWN_OBJECTS = {
    "person": "audios/person.wav",
    "car": "audios/car.wav",
    "backpack": "audios/backpack.wav",
    "bottle": "bottle.wav",
    "cup": "audios/cup.wav",
    "chair": "audios/chair.wav",
    "laptop": "audios/laptop.wav",
    "mouse": "audios/mouse.wav",
    "keyboard": "audios/keyboard.wav",
    "cell phone": "audios/phone.wav"
}

KNOWN_WIDTHS = {
    "person": 0.5,
    "car": 2,
    "backpack" : 0.55,
    "bottle" : 0.20,
    "cup" : 0.15,
    "chair" : 0.50,
    "laptop" : 0.40,
    "mouse" : 0.10,
    "keyboard" : 0.30,
    "cell phone" : 0.15,  
}

# Initialize pygame mixer for audio playback
pygame.mixer.init()

class Detector:
    def __init__(self, server_address, configPath, modelPath, classesPath, focalLength, use_local_camera):
        self.server_address = server_address
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        self.focalLength = focalLength
        self.use_local_camera = use_local_camera

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()
        self.stop_event = threading.Event()
        self.audio_queue = Queue()  # Queue to store objects for audio announcements
        self.queued_objects = set()  # Set to track objects in the audio queue
        self.current_objects = set()  # Track currently detected objects
        self.last_announced_objects = {}  # Dictionary to track last announcement times
        self.objects_within_distance = {}  # Dictionary to store objects within distance

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()

        self.classesList.insert(0, '__Background__')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def play_audio(self):
        while not self.stop_event.is_set():
            if not self.audio_queue.empty():
                classLabel = self.audio_queue.get()
                
                # Play audio if the object is still in the current detection frame
                if classLabel in self.current_objects and classLabel in KNOWN_OBJECTS:
                    audio_file = KNOWN_OBJECTS[classLabel]
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()

                    time.sleep(2)  # Wait 3 seconds before playing the next object

                # Remove the object from the queued_objects set after it has been played
                self.queued_objects.discard(classLabel)
            else:
                time.sleep(0.01)  # Brief sleep to avoid busy-waiting when the queue is empty

    def distance_to_camera(self, known_width, per_width, focal_Length):
        return (known_width * focal_Length) / per_width

    def receive_frames(self):
        """Receive frames from either a local camera or a URL stream."""
        if self.use_local_camera:
            cap = cv2.VideoCapture(self.server_address)  # Use local camera (0 for default camera)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    yield frame
                else:
                    break
            cap.release()
        else:
            stream = requests.get(self.server_address, stream=True)
            bytes_received = bytes()
            for chunk in stream.iter_content(chunk_size=1024):
                bytes_received += chunk
                a = bytes_received.find(b'\xff\xd8')  # JPEG start marker
                b = bytes_received.find(b'\xff\xd9')  # JPEG end marker
                if a != -1 and b != -1:
                    jpg = bytes_received[a:b + 2]
                    bytes_received = bytes_received[b + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    yield frame

    def process_frames(self):
        for frame in self.receive_frames():
            classLabelIDs, confidences, bboxs = self.net.detect(frame, confThreshold=0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold=0.55, nms_threshold=0.2)

            detected_objects = set()
            current_time = time.time()

            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]

                    if classLabel in KNOWN_OBJECTS:
                        detected_objects.add(classLabel)

                    classColor = [int(c) for c in self.colorList[classLabelID]]
                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)

                    x, y, w, h = bbox
                    
                    if classLabel in KNOWN_WIDTHS:
                        width = w
                        distance = self.distance_to_camera(KNOWN_WIDTHS[classLabel], width, focal_Length=self.focalLength)
                        distance_text = "Distance: {:.2f}m".format(distance)
                        if distance < 1:
                            if classLabelID not in self.objects_within_distance:
                                self.objects_within_distance[classLabelID] = current_time
                                print(f"New object detected: {classLabel}")
                            else:
                                if current_time - self.objects_within_distance[classLabelID] >= 1:
                                    self.objects_within_distance[classLabelID] = current_time
                        else:
                            if classLabelID in self.objects_within_distance:
                                del self.objects_within_distance[classLabelID]
                    else:
                        distance_text = "Distance: Unknown"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color=classColor, thickness=1)
                    cv2.putText(frame, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                    cv2.putText(frame, distance_text, (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

            self.update_audio_queue(detected_objects)
            self.current_objects = detected_objects  # Update the set of currently detected objects

            cv2.imshow("Result", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or self.stop_event.is_set():
                break

        cv2.destroyAllWindows()

    def update_audio_queue(self, detected_objects):
        """Add only new objects to the audio queue with a cooldown for re-announcements."""
        current_time = time.time()
        cooldown_time = 5  # 10 seconds cooldown between re-announcements of the same object

        for obj in detected_objects:
            last_announced = self.last_announced_objects.get(obj, 0)
            if obj not in self.queued_objects and current_time - last_announced > cooldown_time:
                self.audio_queue.put(obj)
                self.queued_objects.add(obj)  # Add object to the set to prevent duplicates
                self.last_announced_objects[obj] = current_time  # Update the last announcement time

    def start_processing(self):
        # Start the audio playback thread
        threading.Thread(target=self.play_audio, daemon=True).start()

        # Start the object detection and processing
        threading.Thread(target=self.process_frames, daemon=True).start()

    def stop_processing(self):
        self.stop_event.set()