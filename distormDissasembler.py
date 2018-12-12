#!/usr/bin/env python
"""
# Use distorm3 disaasembler and disaasemble file.
# Return assembly code, instruction list, OpCode list
#
# Author: Necmettin Çarkacı
#
# E-mail: necmettin [ . ] carkaci [ @ ] gmail [ . ] com
#
# Usage :diassembler.py file
#   dataset : file or directory
"""

import sys
import distorm3  # distorm disassembler library
from dissamblerAbstract import disassemblerAbstract
class DistormDisassemblerAbstract(disassemblerAbstract):


    def diassemble(self,filename, bits='32bit'):
        """
            Disassembly executable file return iterable instruction set.

            :param filename : Executable file path
            :type filename: str
            :param bits : File platform 16, 32 or 64.
            :type bits : str [16bit, 32bit, 64bit] (default:32bit)
            :return: assembly code iterator:
            :rtype: iterator
        """
        # Read file content as binary
        with open(filename,'rb') as input_file:
            code = input_file.read()


        # define distorm arguments
        offset = 0

        # define distorm mode 16, 32 or 64 bits
        if bits == '16bit':
            mode = distorm3.Decode16Bits
        elif bits == '32bit':
            mode = distorm3.Decode32Bits
        else:
            mode = distorm3.Decode64Bits

        # Decode assembly instructions
        iterable = distorm3.DecodeGenerator(offset, code, mode)

        return iterable

    def getDisassembledCode(self,filename, delimeter='\n', bits='32bit'):
        """
            Disassemble file and concatenete offset, size, hexcode and instruction into string result.

            :param filename: Binary file name
            :type filename: str
            :param delimeter: Line delimeter for instruction set
            :type delimeter: str
            :param bits: File platform 16, 32 or 64.
            :type bits: str [16bit, 32bit, 64bit] (default:32bit)
            :return assembly instruction list
            :rtype : str
        """
        iterable = self.diassemble(filename,bits)

        diassembled_code = ''
        for (offset, size, instruction, hexdump) in iterable:
            # To avoid TypeError: a bytes-like object is required, not 'str'
            instruction = instruction.decode()
            hexdump     = hexdump.decode()

            diassembled_code += ("%.8x\t%d\t%s\t%s" % (offset, size, hexdump, instruction))+delimeter


        return diassembled_code


    def getAssemblyCode(self,filename, delimeter='\n', bits='32bit'):

        iterable = self.diassemble(filename,bits)

        assembly_code = ''
        for (offset, size, instruction, hexdump) in iterable:
            # To avoid TypeError: a bytes-like object is required, not 'str'
            instruction = instruction.decode()
            assembly_code += str(instruction)+delimeter

        return assembly_code

    def getAssemblyCodeList(self,filename, bits='32bit'):

        iterable = self.diassemble(filename,bits)

        assembly_code_list = []
        for (offset, size, instruction, hexdump) in iterable:
            # To avoid TypeError: a bytes-like object is required, not 'str'
            instruction = instruction.decode()

            assembly_code_list.append(str(instruction))

        return assembly_code_list


