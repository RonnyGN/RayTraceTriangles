import cv2
import numpy as np

def render(img, filepath, use_reducer=False):
    img = img.transpose(1, 0, 2)
    if use_reducer:
        img = img/(1+img)
    img = (img*255).astype(np.uint8)
    display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename=filepath, img=display_img)