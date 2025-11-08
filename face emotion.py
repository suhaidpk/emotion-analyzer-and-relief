import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze emotions in real-time
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        emotions = result[0]['emotion']

        # Display the emotion label
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Optional: show emotion confidence for debugging
        y0 = 90
        for emo, val in emotions.items():
            cv2.putText(frame, f"{emo}: {val:.2f}%", (50, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y0 += 25

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Emotion Detector (DeepFace)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
