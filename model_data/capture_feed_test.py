import cv2

# Open the camera (0 is usually /dev/video0)
cap = cv2.VideoCapture(0)

# Set smaller window resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Create a window
cv2.namedWindow("Pi Camera Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pi Camera Feed", 400, 300)  # small window

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow("Pi Camera Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()
