import cv2


def load_and_resize_image(image_path, resize_img):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")
    img_resized = cv2.resize(img, resize_img)
    return img_resized

def load_images(image_paths, resize_img):
    images = []
    for path in image_paths:
        img_resized = load_and_resize_image(path, resize_img)
        images.append(img_resized)
    return images

def extract_image_patches(img, boxes):
    patches = []
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue
        patches.append(crop)
    return patches