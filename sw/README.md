This folder contains the computer vision software for detecting and decoding videos/images showing the RocSync PCB.

Pipeline:
1. Find ArUco marker
2. Use marker corners to perform coarse homographic reprojection
3. Find corner LEDs in the reprojected image
4. Use LED coordinates to perform accurate reprojection
5. Read circle and binary counter by thresholding given areas
(6. Fit robust linear model to all extracted timestamps and reject outliers)