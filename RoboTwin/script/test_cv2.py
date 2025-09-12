import cv2

def main():
    img = cv2.imread("script/vscode.png")
    print(img.shape)
    cv2.imshow("vscode", img)
    key_pressed = cv2.waitKey(0) & 0xFF
    print("pressed key:", key_pressed)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()