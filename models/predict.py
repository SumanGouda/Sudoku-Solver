import numpy as np


def predict_single_cell(cell, model, pixel_threshold=0.1):
    """
    Predicts a digit for a single 50x50 cell, or returns " " if the cell
    is blank -- blank cells are never passed to the model at all.

    Cell convention (matches extract_and_clean_cells' output):
    ink/digit pixels = 0, background = 1.
    """
    margin = 12
    center_region = cell[margin:-margin, margin:-margin]

    # Fraction of center pixels that are actually ink (value near 0).
    # Ink = 0 here, not 1, so this counts low-value pixels directly
    # instead of averaging -- averaging the raw region would measure
    # background density, not ink density, and flag blanks backwards.
    ink_density = np.mean(center_region < 0.5)

    if ink_density < pixel_threshold:
        return " "

    input_cell = cell.reshape(1, 50, 50, 1).astype('float32')
    prediction = model.predict(input_cell, verbose=0)
    predicted_digit = np.argmax(prediction) + 1  # 9-class model: index 0-8 -> digit 1-9

    return str(predicted_digit)


def predict_sudoku_grid(cleaned_cells, model):
    """
    Runs predict_single_cell over all 81 extracted cells.
    Returns a flat list of 81 items: digit strings "1"-"9", or " " for blank.
    """
    return [predict_single_cell(cell, model) for cell in cleaned_cells]