# cvat_annotation_verification

This repo is automation pipeline to verify if the images are correctly annotated or not

### Configuration

Before running the scripts, make sure to configure your AWS credentials and set appropriate parameters in the `config.yaml` file to match your project's requirements.

### Running the Scripts

To run the entire pipeline, execute the `run_script.sh` script from the root directory:

```bash
./run_script.sh
```

This script automates the following tasks:
1. crops all the objects according to its classname
2. Extracts features in the cropped images using VIT(vision transformer)
3. Runs anamoly detection using isolation forest and lists down all the image name with numbers which might contain mistake

## Scripts and Utilities Description

- **crop_all_segmentation.py**: Crops objects according to its classname and polygon annotations
- **vit_feature_extraction.py**: Extracts features from the cropped images using VIT
- **outlier_detection.py**: Uses isolation forest algorithm to write down all possible wrongly annotated images

## Directory Structure

```
├── config.yaml
├── crop_all.py 
├── crop_all_segmentation.py
├── outlier_detection.py
├── vit_feature_extraction.py
├── run_script.sh
```