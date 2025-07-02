import cv2

# Open the camera
cap = cv2.VideoCapture(0)

# Set resolution (optional but helps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Try adjusting brightness, contrast, and gain (depends on camera support)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)   # range usually 0 to 1
cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
cap.set(cv2.CAP_PROP_GAIN, 0.4)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)      # auto exposure may need to be OFF for this to take effect

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Create a window
cv2.namedWindow("Pi Camera Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pi Camera Feed", 400, 300)

# Create CLAHE object for better visibility in low-light
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    # Convert to LAB color space for CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Display enhanced frame
    cv2.imshow("Pi Camera Feed", frame_eq)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
