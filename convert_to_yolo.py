import os
import shutil
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm

def convert_to_yolo_format(xml_file, label_dir, img_dst_dir, frame_fraction, img_src_dir=os.path.join('data', 'DETRAC-Images')):
    """Convert XML annotations to YOLO format and copy images to the destination directories."""

    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(img_dst_dir, exist_ok=True)
    tree = ET.parse(xml_file)
    root = tree.getroot()
    sequence_name = root.get('name')
    frames = root.findall('frame')
    # Sample a fraction of frames if specified
    # sample every nth frame
    step = int(1 / frame_fraction)
    frames = frames[::step]
    
    for frame in tqdm(frames, desc="Processing frames", position=1, leave=False):
        frame_number = frame.get('num')
        image_id = frame_number.zfill(5)
        image_name = 'img' + image_id + '.jpg'
        image_path = os.path.join(img_src_dir, sequence_name, image_name)
        dst_image_name = f"{sequence_name}_{image_id}.jpg"
        # copy image to destination directory
        dst_image_path = os.path.join(img_dst_dir, dst_image_name)
        if not os.path.exists(dst_image_path):
            shutil.copy(image_path, dst_image_path)
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
        target_list = frame.find('target_list')

        # create label file for the current frame
        label_file_name = f"{sequence_name}_{image_id}.txt"
        if not os.path.exists(os.path.join(label_dir, label_file_name)):
            with open(os.path.join(label_dir, label_file_name), 'w') as f:
                for target in target_list.findall('target'):
                    box = target.find('box')
                    box_left = float(box.get('left'))
                    box_top = float(box.get('top'))
                    box_width = float(box.get('width'))
                    box_height = float(box.get('height'))
                    class_id = 0
                    # Convert to YOLO format (class_id, x_center, y_center, width, height)
                    x_center = (box_left + box_width / 2) / image_width
                    y_center = (box_top + box_height / 2) / image_height
                    width = box_width / image_width
                    height = box_height / image_height
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":

    data_dir = "data"
    xml_dirs = ["DETRAC-Train-Annotations-XML", "DETRAC-Test-Annotations-XML"]
    dataset_name = "DETRAC"
    frame_fraction = 1 # fraction of each sequence's frames to process

    for xml_dir in xml_dirs:
        xml_path = os.path.join(data_dir, xml_dir)
        for xml_file in tqdm(os.listdir(xml_path), desc=f"Processing {xml_dir}", position=0):
            if xml_file.endswith('.xml'):
                label_dir = os.path.join(data_dir, dataset_name, "labels", xml_dir.split('-')[1].lower())
                img_dst_dir = os.path.join(data_dir, dataset_name, "images", xml_dir.split('-')[1].lower())
                convert_to_yolo_format(os.path.join(xml_path, xml_file), label_dir, img_dst_dir, frame_fraction)

    # create dataset yaml file
    dataset_yaml = os.path.join(data_dir, dataset_name, f"{dataset_name}.yaml")
    with open(dataset_yaml, 'w') as f:
        f.write(f"path: /content/{dataset_name}\n")
        f.write(f"train: {'images/train'}\n")
        f.write(f"val: {'images/test'}\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['vehicle']\n")