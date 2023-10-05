#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from ximea import xiapi

# Toggle to enable/disable clipping warning
show_clipping = True

# OpenCV window name
WINNAME = "XiCAM viewer"

# Open video camera
cam_xi = xiapi.Camera()
cam_xi.open_device()
cam_xi.set_exposure(20000)
cam_xi.start_acquisition()

# Create instance of Image to store image data and metadata
img_xi = xiapi.Image()

print("Starting video.")
print("Press c to toggle clip warning.")
print("Press q to exit.")
print(f"Clip warning: {'Enabled' if show_clipping else 'Disabled'}")

try:
    while True:
        cam_xi.get_image(img_xi)
        img_gray = img_xi.get_image_data_numpy()

        # Recolor clipped intensities as full red
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        clipped_idxs = (img_gray == 255).nonzero()
        img_rgb[clipped_idxs] = [0, 0, 255]  # bgr

        cv2.imshow(WINNAME, img_rgb if show_clipping else img_gray)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            show_clipping = not show_clipping
            print(f"Clip warning: {'Enabled' if show_clipping else 'Disabled'}")

        elif key == ord("q"):
            break

except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()
print("Stopping acquisition...")
cam_xi.stop_acquisition()
cam_xi.close_device()
print("Done.")
