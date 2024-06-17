
############################
# Author: Sadegh Naderi
# Date created: 07.02.2024
# Path: ML23-01-Keyword-Spotting-with-an-Arduino-Nano-33-BLE-Sense\report\Code\testExample\testCalc.py
# Version: 1.0
# Reviewed by: Sadegh Naderi
# Review Date: 07.02.2024
############################


#
# @file testCalc.py
# 
# @brief Unit tests for the calc module.
# 
# This module contains unit tests for the functions in the calc module. The tests cover basic arithmetic operations.
# 
# @author Sadegh Naderi
# @date Created: 07.02.2024
# @version 1.0


import unittest
import calc

class TestCalc(unittest.TestCase):
    """! Unit tests for the calc module."""

    def test_addition(self):
        """! Test the addition function in calc module.

        This test case checks the correctness of the addition function
        in the calc module by evaluating it with different input values.

        Test cases:
        1. Positive integers (5, 5): Expected result is 10.
        2. Mixed integers (-1, 1): Expected result is 0.
        """
        self.assertEqual(calc.addition(5, 5), 10)
        self.assertEqual(calc.addition(-1, 1), 0)
