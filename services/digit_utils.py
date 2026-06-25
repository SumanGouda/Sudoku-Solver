import cv2
import numpy as np

def extract_and_clean_cells(warped_image, border_pad=5):
    cleaned_cells = []
    cell_size = 50
    h, w = warped_image.shape
    
    for r in range(9):
        for c in range(9):
            y1, y2 = r * cell_size, (r + 1) * cell_size
            x1, x2 = c * cell_size, (c + 1) * cell_size
            
            cell_gray = warped_image[y1:y2, x1:x2]
            
            center_crop = cell_gray[15:35, 15:35]
            
            # If the center is "flat" (low variance), it is empty.
            if np.std(center_crop) < 10:   
                cleaned_cells.append(np.ones((cell_size, cell_size), dtype=np.uint8))
                continue
                
            # If it's NOT empty, threshold locally
            inner = cell_gray[border_pad:-border_pad, border_pad:-border_pad]
            _, thresh = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            padded = cv2.copyMakeBorder(
                thresh, border_pad, border_pad, border_pad, border_pad,
                cv2.BORDER_CONSTANT, value=255 # Changed from 1 to 255
            )
            
            # (Ink = 0, Background = 1)
            binary_0_1 = (padded // 255).astype(np.uint8)
            cleaned_cells.append(binary_0_1)
            
    return cleaned_cells