'''
    This is a script written to extract frames from videos
    Frames Per Second rate can be controlled
'''

import cv2

# Playing video from file:
cap = cv2.VideoCapture('dataset/')
currentFPS = cap.get(cv2.CAP_PROP_FPS)

currentFrame = 0
n = 0
i = 0

# Change this parameter to a desired number of frames per second
targetFPS = 15

while True:
    ret, frame = cap.read()

    # Three digits title to fit weird OS library sorting
    if (targetFPS * n) % currentFPS == 0:
        if len(str(currentFrame)) == 1:
            name = './data/00' + str(currentFrame) + '.jpg'
            print('Creating...' + name)
            cv2.imwrite(name, frame)
            currentFrame += 1
        elif len(str(currentFrame)) == 2:
            name = './data/0' + str(currentFrame) + '.jpg'
            print('Creating...' + name)
            cv2.imwrite(name, frame)
            currentFrame += 1
        elif len(str(currentFrame)) == 3:
            name = './data/' + str(currentFrame) + '.jpg'
            print('Creating...' + name)
            cv2.imwrite(name, frame)
            currentFrame += 1

    n += 1

    if not ret:
        break

print('Done!')
cap.release()
cv2.destroyAllWindows()
