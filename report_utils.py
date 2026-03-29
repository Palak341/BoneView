import numpy as np

def detect_bone_region(box):
    x1, y1, x2, y2 = box
    y_center = (y1 + y2) / 2

    # Simple vertical segmentation
    if y_center < 200:
        return "Upper limb (shoulder region)"
    elif y_center < 400:
        return "Mid limb (humerus/forearm)"
    else:
        return "Lower limb (wrist/hand region)"


def calculate_fracture_area(box, image_shape):
    x1, y1, x2, y2 = box

    box_area = (x2 - x1) * (y2 - y1)
    total_area = image_shape[0] * image_shape[1]

    return round((box_area / total_area) * 100, 2)