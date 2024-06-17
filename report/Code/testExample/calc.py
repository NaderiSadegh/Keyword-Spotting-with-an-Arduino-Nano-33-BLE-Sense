############################
# Author: Sadegh Naderi
# Date created: 07.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\report\Code\testExample\calc.py
# Version: 1.0
# Reviewed by: Sadegh Naderi
# Review Date: 07.02.2024
############################

##
# @mainpage Simple Addition Testing
#
# @section intro Introduction
#
# This project consists of a simple calculator module (@ref calc) and its corresponding unit tests (@ref testCalc).
# The calculator module provides a basic arithmetic operation, addition.
# The unit tests in testCalc.py ensure the correctness of the calculator function.
#
# @section modules Modules
#
# - @ref calc: Simple calculator module with a basic arithmetic operation.
# - @ref testCalc: Unit tests for the function in the calc module.
#
# @subsection calc_module Calculator Module (calc.py)
#
# The calc module provides the following function:
# - @ref calc.addition: Calculate the addition of two numbers.
#   - Example: `calc.addition(5, 5)` returns `10`.
#
# @subsection test_module Unit Test Module (testCalc.py)
#
# The testCalc module contains the following unit tests:
# - @ref TestCalc.test_addition: Test the addition function in calc module.
#   - Test cases:
#     - Positive integers (5, 5): Expected result is 10.
#     - Mixed integers (-1, 1): Expected result is 0.
#
# @section info Documentation Information
# 
# @author Sadegh Naderi
# @date Created: 07.02.2024
# @version 1.0

##
# @file calc.py
# 
# @brief Simple calculator module.
# 
# This module provides basic arithmetic operations, such as addition, subtraction, multiplication, and division.
# 
# @author Sadegh Naderi
# @date Created: 07.02.2024
# @version 1.0


def addition(x, y):
    """! Calculate the addition of two numbers.

    This function takes two numbers, `x` and `y`, and returns their sum.

    @param x: The first number.
    @param y: The second number.
    @return: The sum of the two numbers.
    """
    return x + y
    