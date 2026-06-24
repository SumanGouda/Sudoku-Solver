class SudokuSolver:
    def __init__(self, board):
        self.sud = [row[:] for row in board]
        
    def is_safe(self, row, col, digit):
        for i in range(9):
            if self.sud[row][i] == digit or self.sud[i][col] == digit:
                return False
        
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if self.sud[i][j] == digit:
                    return False
        return True
    
    def helper(self, row, col):
        if row == 9: return True
        
        next_row = row + 1 if col == 8 else row
        next_col = (col + 1) % 9
        
        # Ensure we treat " " or 0 as empty
        if self.sud[row][col] != 0 and self.sud[row][col] != " ": 
            return self.helper(next_row, next_col)
    
        for digit in range(1, 10):  
            if self.is_safe(row, col, digit):
                self.sud[row][col] = digit
                if self.helper(next_row, next_col): return True
                self.sud[row][col] = 0 # Backtrack
        return False

    def solve(self):
        self.helper(0, 0)
        return self.sud