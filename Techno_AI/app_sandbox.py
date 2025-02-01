import cv2
import pyttsx3
from ultralytics import YOLO

# Initialize pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust the speaking rate

# Load the YOLO model
yolo = YOLO('yolov8s.pt')

# Load the video capture
videoCap = cv2.VideoCapture(0)

def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, 1200, 850))

    results = yolo.track(frame, stream=True)

    spoken_texts = []  # List to store detected objects to be spoken

    for result in results:
        classes_names = result.names

        for box in result.boxes:
            if box.conf[0] > 0.4:
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                colour = getColours(cls)
                confidence = box.conf[0] * 100

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f'{class_name} {confidence:.2f}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
                
                spoken_texts.append(f"{class_name} detected with {confidence:.1f} percent confidence")
    
    # Speak detected objects
    if spoken_texts:
        engine.say(". ".join(spoken_texts))
        engine.runAndWait()
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCap.release()
cv2.destroyAllWindows()

