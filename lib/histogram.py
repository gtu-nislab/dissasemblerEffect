#!/usr/bin/env python
"""
#
# Author: Necmettin Çarkacı
#
# E-mail: necmettin [ . ] carkaci [ @ ] gmail [ . ] com
#
# Usage :opcode.py file
#   dataset : file or directory
"""

from collections import Counter

def calculate_histogram_of_the_text(text, delimeter=','):

    text_list = text.split(delimeter)

    return dict(Counter(text_list))


def getHistogramOfTheFile(filename, delimeter=','):
    """
    Calculate frequency histogram of the given file content.
    It use delimeter as observe different piece of the text

    :param filename: The file path
    :type filename: str
    :param delimeter:The delimeter of the text piece
    :type delimeter:str
    :return:File frequency histogram
    :rtype:dict (text_piece, frequency)
    """
    with open(filename,'r') as input_file:
        content = input_file.read()

    return calculate_histogram_of_the_text(content,delimeter=delimeter)