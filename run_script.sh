#!/bin/bash

# If you have detection(bbox) annotations, then run:
# Run crop_all.py
#echo "Running crop_all.py script..."
#python3 crop_all.py
#echo "Cropping completed"

# If you have segmentation(polygon) annotations, then run:
# Run crop_all_segmentation.py
echo "Running crop_all_segmentation.py script..."
python3 crop_all_segmentation.py
echo "Cropping completed"


# Run vit_feature_extraction.py
echo "Running vit_feature_extraction.py script..."
python3 vit_feature_extraction.py
echo "Feature extraction completed"

# Run outlier_detection_trial.py
echo "Running outlier_detection.py script..."
python3 outlier_detection.py
echo "Outlier detection completed"
