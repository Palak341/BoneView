import cv2
import numpy as np

def generate_gradcam(image, results):
    # Convert image to float [0,1]
    img = image.astype(np.float32) / 255.0

    # Create empty heatmap
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy
        confs = results[0].boxes.conf

        # Convert to numpy safely
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
        if hasattr(confs, "cpu"):
            confs = confs.cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)

            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            blob = np.ones((h, w), dtype=np.float32) * float(conf)

            heatmap[y1:y2, x1:x2] += blob

    # Normalize
    heatmap = np.clip(heatmap, 0, 1)

    # Smooth (Grad-CAM effect)
    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)

    # Apply color
    heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    # Overlay
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return overlay