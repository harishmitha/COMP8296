import sys
import os
import yaml

# Add the yolov5 directory to the PYTHONPATH
sys.path.append(os.path.abspath('yolov5'))  # Adjust if needed

from utils.general import download, Path  # Now this should work

# Load the YAML file
with open('/home/comp8296/Desktop/Object/Object_Detection_Files/coco.yaml', 'r') as f:
    yaml_data = yaml.safe_load(f)

# Set the download variables
segments = False  # Set to True if you want segment labels
dir = Path(yaml_data['path'])  # Dataset root directory
url = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/'

# Download labels
urls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # Labels
download(urls, dir=dir.parent)

# Download data
urls = [
    'http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images
    'http://images.cocodataset.org/zips/val2017.zip',    # 1G, 5k images
    'http://images.cocodataset.org/zips/test2017.zip'    # 7G, 41k images (optional)
]
download(urls, dir=dir / 'images', threads=3)
