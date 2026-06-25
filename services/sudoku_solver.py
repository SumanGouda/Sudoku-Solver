class SudokuSolver:
    def __init__(self, board): 
        self.board = [[int(cell) for cell in row] for row in board]
        self.is_valid_start = self._validate_initial_board()

    def _validate_initial_board(self):
        """Checks the board for existing violations and returns False if invalid."""
        for r in range(9):
            for c in range(9):
                val = self.board[r][c]
                if val != 0: 
                    self.board[r][c] = 0
                    if not self._is_safe(r, c, val):
                        print(f"VIOLATION FOUND at ({r}, {c}) with value {val}")
                        return False
                    self.board[r][c] = val
        return True

    def _is_safe(self, row, col, digit): 
        # Check Row and Column
        for i in range(9):
            if self.board[row][i] == digit or self.board[i][col] == digit:
                return False
         
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if self.board[i][j] == digit:
                    return False
        return True

    def solve(self): 
        if not self.is_valid_start:
            return False
        return self._backtrack(0, 0)

    def _backtrack(self, row, col): 
        if row == 9:
            return True
        
        next_row, next_col = (row, col + 1) if col < 8 else (row + 1, 0)
        
        if self.board[row][col] != 0:
            return self._backtrack(next_row, next_col)
    
        for digit in range(1, 10):
            if self._is_safe(row, col, digit):
                self.board[row][col] = digit
                if self._backtrack(next_row, next_col):
                    return True
                self.board[row][col] = 0  # Backtrack
        return False