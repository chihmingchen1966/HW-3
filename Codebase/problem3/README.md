
# Homework 3 — Problem 3

This folder contains my solution for **Homework 3, Problem 3**.

## Overview

Problem 3 focuses on implementing and testing the required algorithm or computational procedure for the third part of Homework 3.

The purpose of this folder is to provide a clean and reproducible implementation so that the result discussed in the final report can be verified directly from the submitted code.

The implementation is organized to:

* execute the required Problem 3 experiment,
* generate the result used in the final PDF report,
* keep the Problem 3 code independent from the other homework problems,
* make the execution process simple and reproducible for the grader.

## Files

```text
problem3/
├── README.md
└── problem3_main.py
```

### `problem3_main.py`

Main Python script for Problem 3.

This script contains the implementation used to generate the output reported in the HW3 Problem 3 final report.

Depending on the assignment requirement, the script may include:

* initialization of the problem setting,
* construction of the required model or objective function,
* execution of the algorithm,
* printing of the final result,
* optional verification or diagnostic output.

### `README.md`

This file.

It explains the purpose of the folder, the expected file structure, and the basic command for running the script.

## How to Run

From the root directory of the repository, run:

```bash
python3 Codebase/problem3/problem3_main.py
```

Or, if you are already inside this folder:

```bash
python3 problem3_main.py
```

## Expected Output

After running the script, the terminal should display the main result for Problem 3.

The output is intended to match the result, analysis, and discussion presented in the final PDF report for Homework 3, Problem 3.

If the script prints intermediate values, they are included to make the computation easier to inspect and verify.

## Reproducibility

The code is designed to be reproducible.

If the experiment uses randomness, the random seed is fixed inside the script or provided through command-line arguments. This helps ensure that the same result can be obtained when the grader reruns the program.

## Notes for Grading

* The main script is `problem3_main.py`.
* The expected way to run the code is shown in the **How to Run** section.
* The final numerical result or algorithm output should correspond to the result reported in the HW3 Problem 3 PDF.
* This folder only contains files related to Problem 3, so the grader can inspect this problem independently.

## Relationship to the Report

The Python script provides the computational result.

The corresponding final report provides:

* the problem background,
* the mathematical formulation,
* the explanation of the method,
* the interpretation of the output,
* and the final conclusion for Problem 3.

Therefore, this folder should be read together with the HW3 Problem 3 final report PDF.
