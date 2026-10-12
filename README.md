# Vehicle Tracking by Detection
In this project real-time vehicle tracking by detection is implemented using YOLO [^1] and SORT [^2]. For this the SORT (Simple Online and Realtime Tracking) algorithm is implemented in Python. SORT predicts the movement of tracked objects over a sequence of images with a Kalman Filter. Detections from an object detection model are used to correct these predictions. The detections are matched to tracked objects based on IoU (Intersection over Union) using the Hungarian Algorithm. 

A yolo26s model is trained on the UA-DETRAC [^3] dataset to provide vehicle detections. the YOLO (You Only Look Once) family of models are real-time capable object detection models. They are single stage detectors, meaning they predict the bounding boxes and classes of objects in a single pass of the network. The UA-DETRAC dataset is a large dataset for vehicle detection, it consists of 100 challenging video sequences of real-world traffic scenes with over 140,000 image frames.

## Usage

`convert_to_yolo.py`: Converts UA-DETRAC annotations into YOLO format and copies the images and labels into train and test folders for training. You can download the images and annotations [here](https://sites.google.com/view/daweidu/projects/ua-detrac?authuser=0).

`train_yolo_detector.ipynb`: Trains a YOLO model for vehicle detection in google colab environment to make use of GPU runtime provided by Google. Make sure to upload the zipped data folder to google drive and select a GPU runtime in colab.

`tracking.py`: Real-time vehicle tracking on an image sequence with a trained YOLO detector and SORT.

## Example Video

### References
[^1]: G. Jocher, J. Qiu, M. Liu, S. Lyu, Akyon, Fatih Cagatay, and Kalfaoglu, Muhammet Esat. Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models. arXix. DOI: [10.48550/arXiv.2606.03748](https://doi.org/10.48550/arXiv.2606.03748)
[^2]: Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple online and realtime tracking. arXiv, DOI: [10.48550/arXiv.1602.00763](https://doi.org/10.48550/arXiv.1602.00763)
[^3]: Wen, L., Du, D., Cai, Z., Lei, Z., Chang, M.-C., Qi, H., Lim, J., Yang, M.-H., & Lyu, S. (2015). UA-DETRAC: A New Benchmark and Protocol for Multi-Object Detection and Tracking (Version 4). arXiv. DOI: [10.48550/ARXIV.1511.04136](https://doi.org/10.48550/ARXIV.1511.04136)
