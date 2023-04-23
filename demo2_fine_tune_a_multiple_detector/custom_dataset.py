# 导入所需库
import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bndbox = root.find("object").find("bndbox")
    xmin, ymin, xmax, ymax = [int(bndbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]
    return xmin, ymin, xmax, ymax

class CustomDataset(Dataset):
    def __init__(self, images_folder, labels_folder, transform):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.transform = transform
        self.image_files = glob.glob(os.path.join(images_folder, "*.jpg"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = os.path.join(self.labels_folder, os.path.basename(image_path).replace(".jpg", ".xml"))
        image = Image.open(image_path).convert("RGB")
        xmin, ymin, xmax, ymax = parse_xml(label_path)
        target = {
            "labels": torch.tensor([1], dtype=torch.long),
            "boxes": torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32),
        }
        image = self.transform(image)
        return {"image": image, "target": target}
