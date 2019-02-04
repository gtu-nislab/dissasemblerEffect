import sys

if sys.version_info[0] >= 3:
    raise Exception("Must be using Python 2")

from unpyclib import disasm
from dissamblerAbstract import disassemblerAbstract

class UnpycDisassembler(disassemblerAbstract):
    def diassemble(self, filename, bits):
        """

        :param filename:
        :param bits:
        :return:
        """

        # Read file content as binary
        with open(filename, 'rb') as input_file:
            code = input_file.read()

        asm = disasm.Disassembler.disasmCommands(code)
        return asm
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

        return self.diassemble(filename,bits).replace("\n",delimeter)

    def getAssemblyCode(self,filename, delimeter='\n', bits='32bit'):

        return self.diassemble(filename,bits).replace("\n",delimeter)


    def getAssemblyCodeList(self,filename, bits='32bit'):

        return self.diassemble(filename,bits).split("\n")


fln = "/home/nislab2/Desktop/DissamblerEffect/benign/0b5511674394666e9d221f8681b2c2e6.exe"
unpy = UnpycDisassembler()
print(unpy.diassemble(fln,bits="32bit"))