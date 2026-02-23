import sys 
import os
from pathlib import Path
sys.path.insert(0, os.path.normcase(Path(__file__).resolve().parents[1]))

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from rocsync.vision import CameraType
from rocsync.vision import ir_corners, corner_dots, aruco_corners_coords, board_size
from rocsync.vision import find_corners_aruco
from rocsync.vision import find_corners_dots
from rocsync.vision import read_ring, read_counter
from rocsync.vision import process_frame

def test_piecewise():
    """ test the individual elements of the process_frame function with the new led layout """
    
    TEST_DIR = Path(__file__).resolve().parents[0]
    
    # prepare the pcb ----------------------------------------------------------
    image = cv.imread(TEST_DIR/"img1.jpg")
    
    aruco_corners = find_corners_aruco(
        image,
        frame_number=999, 
        debug_dir=TEST_DIR/"output_piecewise"
    )
    red_channel = image[:, :, 2]
    rough_transformation_matrix = cv.getPerspectiveTransform(
        aruco_corners, aruco_corners_coords
    )
    rough_pcb = cv.warpPerspective(
        red_channel, rough_transformation_matrix, (board_size, board_size)
    )
    cv.imwrite(TEST_DIR/"output_piecewise"/"rough_pcb.jpg", rough_pcb)

    # still works, additional led is just ignored with the original 4 points. Add the 5th red one just to be safe
    corners = find_corners_dots(
        rough_pcb, 
        999, 
        debug_dir=TEST_DIR/"output_piecewise"
    )
    
    # however, this only works with exactly 4 points
    transformation_matrix = np.dot(
        cv.getPerspectiveTransform(corners[1:, :], corner_dots[1:, :]),
        rough_transformation_matrix,
    )
    pcb = cv.warpPerspective(red_channel, transformation_matrix, (board_size, board_size))
    
    cv.imwrite(TEST_DIR/"output_piecewise"/"pcb.jpg", pcb)
    
    # decode the ring (should remain the same) ---------------------------------
    ring = read_ring(pcb, camera_type=CameraType.RGB, draw_result=True)
    print(f"ring decoded: {ring}")
    
    # decode clock (must be adjusted to new layout) ----------------------------
    counter = read_counter(pcb, camera_type=CameraType.RGB, draw_result=True)
    print(f"decoded clock: {counter} [0.1s]")

    # show the debug image output
    plt.imshow(pcb)
    plt.show()
    
def test_full():
    """ test the full process_frame function for validation. """
    
    TEST_DIR = Path(__file__).resolve().parents[0]
    
    # prepare the pcb ----------------------------------------------------------
    image = cv.imread("C:/Users/steie/code/calibration/RocSync/sw/rocsync/tests/img1.jpg")
     
    out = process_frame(
        image, 
        camera_type=CameraType.RGB,
        frame_number=999,
        debug_dir=TEST_DIR/"output_full"
    )
    print(f"output of process frame was: {out}")

if __name__ == "__main__":
    os.system("cls" if os.name=="nt" else "clear")
    test_full()



    