import cv2

refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        print(f"Selected area: {refPt}")

# 이미지 불러오기
image = cv2.imread("./images/home_2/frame_0000.jpg")
clone = image.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_and_crop)

while True:
    cv2.imshow("Image", clone)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # 'q'를 누르면 종료
        break

cv2.destroyAllWindows()
