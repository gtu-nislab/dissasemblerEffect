#!/usr/bin/env python
"""
#
# Author: Necmettin Çarkacı
#
# E-mail: necmettin [ . ] carkaci [ @ ] gmail [ . ] com
#
# Usage :histogram.py file
#   dataset : file or directory
"""

from lib import histogram

def getHistogram(filename, delimeter=','):
    """
    Calculate opcode frequency histogram
    :param filename:
    :param delimeter:
    :return:
    """
    return histogram.getHistogramOfTheFile(filename, delimeter=delimeter)