# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 23:35:18 2021

@author: leona
"""

import cv2 as cv
import numpy as np


dst = cv.Canny(src, 50, 200, None, 3)

cdst = cvtColor(dst, cv.COLOR_GRAY2BGR)