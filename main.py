import cv2
import winsound
# init camera
cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, image1 = cam.read()
    ret, image2 = cam.read()

    diff = cv2.absdiff(image1, image2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contour, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image1, contour, -1, (0, 255, 0), 2)
    for c in contour:
        if cv2.contourArea(c) < 6000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        winsound.Beep(500, 300)
    cv2.imshow('Khan Camera', image1)
    if cv2.waitKey(1) == ord('q'):  # wait for 10ms for wait-key
        break

cv2.destroyAllWindows()



