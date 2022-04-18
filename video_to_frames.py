import cv2

# script for generating images to each frame from video
if __name__ == '__main__':
    # get vid to convert to frames
    vidcap = cv2.VideoCapture('bsc-3m/09-06-2021-mpeg4-360p-1h.m4v')

    # read first frame
    success, image = vidcap.read()

    # count for keeping track of frame count
    count = 0

    # while there are frames to be read
    while success:
        print(f'reading frame {count}')

        # rescale image from 640x360 to 1280x720
        image = cv2.resize(image, (0, 0), fx=2, fy=2)
        cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file

        # next image and iterate
        success, image = vidcap.read()
        count += 1
