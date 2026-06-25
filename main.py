import io
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from PIL import Image 
from services.image_ops import process_image
from services.digit_utils import extract_and_clean_cells
from models.predict import predict_sudoku_grid
from services.sudoku_solver import SudokuSolver

app = FastAPI()

MODEL = load_model("models/emnist_digit_cnn.keras")


@app.post("/predict")
async def run_sudoku_pipeline(file: UploadFile = File(...)):
    """
    Processes image -> Predicts digits -> Solves the grid -> Returns result.
    """
    try: 
        try:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="Could not decode image.")
 
        p_img = process_image(img)
        cleaned_cells = extract_and_clean_cells(p_img)
        sudoku_predictions = predict_sudoku_grid(cleaned_cells, MODEL)
  
        grid_9x9 = []
        for i in range(9):
            row = []
            for j in range(9):
                val = sudoku_predictions[i*9 + j]
                row.append(int(val) if isinstance(val, (int, str)) and str(val).isdigit() else 0)
            grid_9x9.append(row)
  
        print(grid_9x9)
  
        solver = SudokuSolver(grid_9x9) 
         
        if solver.solve(): 
            flat_solved = [int(item) for row in solver.board for item in row]
            return JSONResponse(content={"solved_grid": flat_solved})
        else: 
            return JSONResponse(
                content={"error": "The puzzle is unsolvable. Check for digit recognition errors."}, 
                status_code=400
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)