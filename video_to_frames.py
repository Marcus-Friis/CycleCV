import cv2


vidcap = cv2.VideoCapture('bsc-3m/09-06-2021-mpeg4-360p-1h.m4v')
success, image = vidcap.read()
count = 0
while success:
    print(f'reading frame {count}')
    image = cv2.resize(image, (0, 0), fx=2, fy=2)
    cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    count += 1
