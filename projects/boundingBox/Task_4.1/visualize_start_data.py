import os
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


images_folder = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/potholes/images"
annotations_folder = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/potholes/annotations"

image_files = sorted([f for f in os.listdir(images_folder) if f.endswith((".jpg", ".png"))])

for img_name in image_files[:5]:
    img_path = os.path.join(images_folder, img_name)
    xml_path = os.path.join(annotations_folder, os.path.splitext(img_name)[0] + ".xml")

    img = Image.open(img_path)
    
    #Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis("off")
    plt.show()
