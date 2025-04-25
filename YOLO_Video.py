from ultralytics import YOLO
import cv2
import os
import pygame

# Initialize pygame mixer for sound alerts
pygame.mixer.init()
alert_sound_path = "alert.mp3"  # Add an MP3 file in same folder
if not os.path.exists(alert_sound_path):
    print("❌ Alert sound not found! Please place 'alert.mp3' in the project directory.")

def video_detection(path_x):
    print(f"🎥 Input Value: {path_x}, Type: {type(path_x)}")

    # Validate webcam or file input
    if isinstance(path_x, int):
        print("📸 Using webcam input.")
    elif isinstance(path_x, str):
        if not os.path.isfile(path_x):
            raise FileNotFoundError(f"❌ File not found: {path_x}")
    else:
        raise TypeError("⚠️ Invalid input type. Expected int (webcam) or str (video path).")

    cap = cv2.VideoCapture(path_x)
    if not cap.isOpened():
        raise ValueError(f"❌ Unable to open video source: {path_x}")
    print("✅ Video source opened successfully.")

    model = YOLO("ppe.pt")
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
                  'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

    while True:
        success, img = cap.read()
        if not success:
            print("🚫 No frame captured. Exiting.")
            break

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                class_id = int(box.cls[0])
                class_name = classNames[class_id]
                label = f"{class_name} {conf}"

                # Color coding
                if class_name in ['Mask', 'Hardhat', 'Safety Vest']:
                    color = (0, 255, 0)
                elif class_name in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                    color = (0, 0, 255)
                elif class_name in ['machinery', 'vehicle']:
                    color = (0, 149, 255)
                else:
                    color = (85, 45, 255)

                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Play alert if NO-Mask is detected
                    if class_name == 'NO-Mask' and os.path.exists(alert_sound_path):
                        pygame.mixer.music.load(alert_sound_path)
                        pygame.mixer.music.play()

        cv2.imshow("PPE Detection", img)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the video detection using default webcam (0)
video_detection(0)
