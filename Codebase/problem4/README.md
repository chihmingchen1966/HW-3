
# Homework 3 — Problem 4

This folder contains my solution for **Homework 3, Problem 4**.

## Overview

Problem 4 focuses on the computational implementation required for the fourth part of Homework 3.

The purpose of this folder is to provide a clear and reproducible implementation so that the result presented in the final report can be verified directly from the submitted code.

The implementation is organized to:

* define the required Problem 4 setting,
* implement the main computational procedure,
* generate the result used in the HW3 Problem 4 final report,
* keep the code for Problem 4 separate from the other homework problems,
* allow the grader to reproduce the output with a simple command.

## Files

```text
problem4/
├── README.md
└── problem4_main.py
```

### `problem4_main.py`

Main Python script for Problem 4.

This script contains the implementation used to produce the output reported in the HW3 Problem 4 final report.

Depending on the assignment requirement, the script may include:

* problem initialization,
* construction of the required model, Hamiltonian, circuit, or objective function,
* execution of the numerical or quantum-inspired algorithm,
* output of the final result,
* optional intermediate information for checking correctness.

### `README.md`

This file.

It explains the purpose of the folder, the expected file structure, and the command used to run the script.

## How to Run

From the root directory of the repository, run:

```bash
python3 Codebase/problem4/problem4_main.py
```

Or, if you are already inside this folder:

```bash
python3 problem4_main.py
```

## Expected Output

After running the script, the terminal should display the main result for Problem 4.

The printed result is intended to correspond to the numerical output, discussion, and conclusion shown in the HW3 Problem 4 final report.

If the script prints additional intermediate values, they are included to make the calculation easier to inspect and verify.

## Reproducibility

The code is designed to be reproducible.

If randomness is involved, the random seed is fixed in the script or specified through command-line arguments. This helps ensure that the grader can obtain the same or consistent results when rerunning the program.

## Notes for Grading

* The main entry point is `problem4_main.py`.
* The recommended execution command is provided in the **How to Run** section.
* The final output should match the result described in the corresponding HW3 Problem 4 PDF report.
* This folder only contains files related to Problem 4, so it can be checked independently from the other homework problems.

## Relationship to the Report

The Python script provides the computational result.

The corresponding final PDF report provides:

* the problem background,
* the mathematical or algorithmic formulation,
* the explanation of the implementation,
* the interpretation of the output,
* and the final conclusion for Problem 4.

Therefore, this folder should be read together with the HW3 Problem 4 final report PDF.
