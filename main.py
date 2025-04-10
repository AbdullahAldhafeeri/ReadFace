import cv2
import os
from deepface import DeepFace

# تحميل صور الوجوه المعروفة
known_faces_dir = "known_faces"
known_faces = []

for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(known_faces_dir, filename)
        name = os.path.splitext(filename)[0]
        known_faces.append({"path": path, "name": name})

# فتح الكاميرا
cap = cv2.VideoCapture(0)

print("اضغط على Q للخروج...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # التعرف على الوجه في الإطار الحالي
        for known in known_faces:
            result = DeepFace.verify(frame, known["path"], enforce_detection=False)
            if result["verified"]:
                cv2.putText(frame, f"Matched: {known['name']}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break
        else:
            cv2.putText(frame, "Unknown", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Face Recognition - DeepFace", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
