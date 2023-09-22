#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import IntEnum


class FFT_LIBS(IntEnum):
    PYFFTW = 0
    ROCKETFFT = 1
    # SCIPY = 2


FFT_LIB = FFT_LIBS.ROCKETFFT
