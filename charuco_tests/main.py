import cv2

# ------------------ Setup board and detector ------------------
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
board = cv2.aruco.CharucoBoard((8, 10), 0.048, 0.036, dictionary)

params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
detector = cv2.aruco.ArucoDetector(dictionary, params)
charuco_detector = cv2.aruco.CharucoDetector(board)

# ------------------ Open webcam (index 0) --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera. Check index or permissions.")
    exit()

print("✅ Camera opened. Press 'q' in the display window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None:
        continue

    # Interpolate Charuco corners
    charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board, None, None
    )

    # print(charuco_retval, charuco_corners, charuco_ids)

    a, b, c, d = charuco_detector.detectBoard(gray)
    print(a, b)

    # Draw results on the original frame
    debug_frame = frame.copy()
    cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids, borderColor=(0, 255, 0))
    if charuco_retval:
        cv2.aruco.drawDetectedCornersCharuco(
            debug_frame, charuco_corners, charuco_ids, (0, 0, 255)
        )

    # Show the live feed
    cv2.imshow("Charuco Detection", debug_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
