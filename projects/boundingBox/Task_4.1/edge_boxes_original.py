import cv2
import os
import numpy as np

images_folder = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/potholes/images"
output_folder = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/proposals"
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(images_folder) if f.lower().endswith((".jpg", ".png"))]

# Initialize EdgeBoxes
edge_detector = cv2.ximgproc.createStructuredEdgeDetection("/Users/dani/Desktop/IDL_project4/projects/boundingBox/model.yml")
edge_boxes = cv2.ximgproc.createEdgeBoxes()
edge_boxes.setMaxBoxes(1000)

for img_name in image_files:
    img_path = os.path.join(images_folder, img_name)
    img = cv2.imread(img_path)

   
    img_resized = cv2.resize(img, (600, 600))

    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    edges = edge_detector.detectEdges(rgb)
    orientation = edge_detector.computeOrientation(edges)
    edges = edge_detector.edgesNms(edges, orientation)

    # EdgeBoxes proposals
    boxes, scores = edge_boxes.getBoundingBoxes(edges, orientation)
    # boxes = [[x, y, w, h], ...]

    np.save(os.path.join(output_folder, img_name + ".npy"), np.array(boxes))

    vis = img_resized.copy()
    for (x, y, w, h) in boxes[:50]:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

    cv2.imwrite(os.path.join(output_folder, img_name), vis)


