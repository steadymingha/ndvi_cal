import cv2

frame = cv2.imread("output/20250812141216_RGB.jpg")
print(frame.mean(axis=0).mean(axis=0))
frame = cv2.imread("output/20250812141215_NOIR.jpg")
print(frame.mean(axis=0).mean(axis=0))