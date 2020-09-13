import cv2
import numpy as np

coords = []
clicked = False
data = []


def on_click(event, x, y, flags, params):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        coords.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE and clicked:
        coords.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        clicked = False
        coords.append((-1, -1))


def main():
    cv2.namedWindow("Draw")
    cv2.setMouseCallback("Draw", on_click)

    while True:
        global coords
        key = cv2.waitKey(1)
        frame = np.ones((280, 280))
        for i in range(1, len(coords)):
            if coords[i-1] != (-1, -1) and coords[i] != (-1, -1):
                cv2.line(frame, coords[i-1], coords[i], (0, 0, 0), 20)
        cv2.imshow("Draw", frame)
        if key == 27:
            break
        elif key == ord('c'):
            coords = []
        elif key == ord('s'):
            inp = cv2.resize(frame, (28, 28))
            inp = 1-inp
            cv2.imshow("Network Input", inp)
            inp = np.resize(inp, (28, 28, 1))
            data.append(inp)
            print(len(data))
            cv2.imshow("Draw", frame)
            cv2.waitKey(0)
            coords = []

    cv2.destroyAllWindows()

    print("Saving Data...")

    bluetooth_count = int(input())
    b_count = int(input())
    usb_count = int(input())
    labels = np.zeros(bluetooth_count+b_count+usb_count)
    labels[bluetooth_count:bluetooth_count+b_count] = 1
    labels[bluetooth_count+b_count:] = 2
    np.save("data/x.npy", np.array(data))
    np.save("data/y.npy", labels)


if __name__ == '__main__':
    main()
