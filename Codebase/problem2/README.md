
# Homework 3 — Problem 2

This folder contains my solution for **Homework 3, Problem 2**.

## Overview

Problem 2 focuses on implementing the required computational experiment for the second part of Homework 3.
The purpose of this folder is to provide a clear and reproducible codebase so that the result shown in the final report can be generated again from the submitted Python script.

The implementation is organized to:

* run the required Problem 2 calculation,
* produce the numerical result or experiment output used in the report,
* keep the code separate from the other homework problems,
* make the execution process easy for the grader to verify.

## Files

```text
problem2/
├── README.md
└── problem2_main.py
```

### `problem2_main.py`

Main Python script for Problem 2.

This script contains the implementation used to generate the output reported in the HW3 Problem 2 final report.

### `README.md`

This file.

It explains the purpose of this folder, the expected file structure, and the basic command used to run the code.

## How to Run

From the root directory of the repository, run:

```bash
python3 Codebase/problem2/problem2_main.py
```

Or, if you are already inside this folder:

```bash
python3 problem2_main.py
```

## Expected Output

After running the script, the terminal should display the main result for Problem 2.

The output is intended to correspond to the result, discussion, and interpretation presented in the final PDF report for Homework 3, Problem 2.

## Reproducibility

The code is written so that the grader can reproduce the submitted result with a simple command.
If a random seed is used in the script, the seed is fixed inside the implementation or specified in the command-line arguments so that the result can be reproduced consistently.

## Notes

* Each homework problem is placed in its own folder for clarity.
* The code in this folder is focused only on Problem 2.
* The full explanation, mathematical background, and result discussion are provided in the corresponding HW3 Problem 2 final report.
* This README is intended to help the grader quickly understand what the folder contains and how to run the script.
