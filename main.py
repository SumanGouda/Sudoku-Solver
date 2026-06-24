import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from services.image_ops import process_image
from services.digit_utils import extract_and_clean_cells
from models.predict import predict_single_cell
from services.sudoku_solver import SudokuSolver

app = FastAPI()

MODEL = load_model("models/sudoku_digit_model.keras")


@app.post("/predict")
async def run_sudoku_pipeline(file: UploadFile = File(...)):
    """
    Processes image -> Predicts digits -> Solves the grid -> Returns result.
    """
    try:
        contents = await file.read()

        try:
            img = Image.open(io.BytesIO(contents))
        except Exception:
            raise ValueError("Could not decode image.")

        # 1. Vision Pipeline
        p_img = process_image(img)
        cleaned_cells = extract_and_clean_cells(p_img)

        # Run every one of the 81 cells through the model (or " " if blank)
        sudoku_predictions = [predict_single_cell(cell, MODEL) for cell in cleaned_cells]

        # 2. Logic Pipeline — convert "1".."9" / " " into 0/ints for the solver
        numerical_board = [0 if x == " " else int(x) for x in sudoku_predictions]
        grid_9x9 = [numerical_board[i*9:(i+1)*9] for i in range(9)]

        print("--- Predicted Grid ---")
        for row in grid_9x9:
            print(row)
        print("----------------------")

        # 3. Solver
        solver = SudokuSolver(grid_9x9)
        if solver.helper(0, 0):
            solved_grid = solver.sud
            flat_solved = [item for row in solved_grid for item in row]
            return JSONResponse(content={"solved_grid": flat_solved})
        else:
            return JSONResponse(content={"error": "No solution found for the provided grid."}, status_code=422)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)