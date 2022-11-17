# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:02:32 2022

@author: Yue
"""

import math


def compute_ELP(IOL_model, AL, RAC):
    A_cons = {"1": 119, "2": 119.1, "3": 119.1}

    ACDcons = 0.62467 * A_cons[IOL_model] - 68.747
    L_COR = AL
    if AL > 24.2:
        L_COR = -3.446 + 1.715 * AL - 0.0237 * AL**2
    K = 337.5 / RAC
    Cw = -5.41 + 0.58412 * L_COR + 0.098 * K
    H = RAC - math.sqrt(RAC**2 - Cw**2 / 4)
    offset = ACDcons - 3.336
    ELP = H + offset
    return ELP
