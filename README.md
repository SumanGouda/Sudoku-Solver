---
title: Sudoku Solver
emoji: 🧩
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Sudoku Solver

A full-stack Sudoku solver: takes a photo of a printed Sudoku puzzle, detects the grid, classifies digits with a CNN trained on EMNIST, solves it with a backtracking algorithm, and returns the solved grid.

## How it works
1. Grid detection and cell extraction via OpenCV
2. Digit classification via a CNN trained on EMNIST digits
3. Solving via a backtracking algorithm
4. Result displayed on the frontend