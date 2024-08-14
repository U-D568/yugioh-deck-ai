import cv2

def main():
    img = cv2.imread("screenshot.jpg")
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (3, 3))
    canny_edges = cv2.Canny(gray_blur, 150, 250)
    cv2.imshow("test", canny_edges)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()