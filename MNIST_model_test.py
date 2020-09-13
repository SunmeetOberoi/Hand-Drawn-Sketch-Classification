import cv2
import numpy as np
import tensorflow as tf

coords = []
clicked = False

model = tf.keras.models.load_model("models/MNIST.h5")


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
        elif key == ord('d'):
            inp = cv2.resize(frame, (28, 28))
            inp = 1-inp
            cv2.imshow("Network Input", inp)
            inp = np.resize(inp, (1, 28, 28, 1))
            res = model.predict(inp)[0]
            prediction = str(np.argmax(res))
            confidence = int(res[int(prediction)]*10000)/100
            if confidence>98.0:
                cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
            else:
                cv2.putText(frame, "Maybe " + prediction, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
            print(prediction, confidence)
            cv2.imshow("Draw", frame)
            cv2.waitKey(0)
            coords = []

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
