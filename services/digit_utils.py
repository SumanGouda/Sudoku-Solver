import cv2
import numpy as np

def extract_and_clean_cells(warped_image, border_pad=4):
    """
    Takes the 450x450 warped_image, extracts 81 cells, 
    crops borders, binarizes, and returns a list of 81 (50x50) 2D arrays.
    """
    cleaned_cells = []
    
    # 1. Image is already 450x450 from process_image
    # Each cell is exactly 50x50
    cell_size = 50
    
    for r in range(9):
        for c in range(9):
            # Extract the raw cell
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            cell = warped_image[y1:y2, x1:x2]
             
            # Slicing removes the outer 'border_pad' pixels from all four sides
            cropped = cell[border_pad:-border_pad, border_pad:-border_pad]
            
            # Resize back to 50x50 after cropping 
            resized = cv2.resize(cropped, (50, 50), interpolation=cv2.INTER_AREA)
            
            # Binarize (Otsu) 
            _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
             
            binary_0_1 = (thresh // 255).astype(np.uint8)
            
            cleaned_cells.append(binary_0_1)
            
    return cleaned_cells