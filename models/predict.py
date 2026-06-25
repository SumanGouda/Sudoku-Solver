import numpy as np

def predict_single_cell(cell, model, pixel_threshold=0.1):
    margin = 12
    center_region = cell[margin:-margin, margin:-margin]

    ink_density = np.mean(center_region < 0.5)

    if ink_density < pixel_threshold:
        return 0  # blank cell

    input_cell = cell.reshape(1, 50, 50, 1).astype('float32')
    prediction = model.predict(input_cell, verbose=0)
    predicted_digit = int(np.argmax(prediction))  

    return predicted_digit

def predict_sudoku_grid(cleaned_cells, model): 
    return [predict_single_cell(cell, model) for cell in cleaned_cells]