import cv2
import os

# Paths
images_folder = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/potholes/images"
output_folder = "/Users/dani/Desktop/IDL_project4/projects/boundingBox/proposals"
os.makedirs(output_folder, exist_ok=True)


image_files = [f for f in os.listdir(images_folder) if f.endswith((".jpg", ".png"))]

for img_name in image_files:
    img_path = os.path.join(images_folder, img_name)
    img = cv2.imread(img_path)

    img = cv2.resize(img, (600, 600))

    # Selective search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()

    rects = ss.process()  # returns list of [x, y, w, h]

    # Draw first 50 proposals
    for (x, y, w, h) in rects[:50]:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #Save images with proposals
    cv2.imwrite(os.path.join(output_folder, img_name), img)
