# test1.py
from capstone import *
from dissamblerAbstract import disassemblerAbstract
class CapstoneDisasembler(disassemblerAbstract):

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
        with open(filename, "rb") as malbyte:
            data = malbyte.read()
            # define arguments
            # define mode 16, 32 or 64 bits
            if bits == '16bit':
                mode = CS_MODE_16
            elif bits == '32bit':
                mode = CS_MODE_32
            else:
                mode = CS_MODE_64

            # Decode assembly instructions
            cpstn = Cs(CS_ARCH_X86, mode)
            cpstn.skipdata=True
            assemblyiter = cpstn.disasm(data, 0x1000)
            return assemblyiter

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

        assembly_code = ''
        for i in iterable:
            # To avoid TypeError: a bytes-like object is required, not 'str'
            (address, mnemonic, op_str) = i.address, i.mnemonic, i.op_str
            assembly_code += ("0x%s:\t%s\t%s"% (address, mnemonic, op_str))+delimeter

        print("filename:" + filename + "\ncode:\n" + assembly_code)
        return assembly_code

    def getAssemblyCode(self,filename, delimeter='\n', bits='32bit'):

        iterable = self.diassemble(filename,bits)

        assembly_code = ''
        for i in iterable:
            # To avoid TypeError: a bytes-like object is required, not 'str'
            (address, mnemonic, op_str) = i.address, i.mnemonic, i.op_str
            assembly_code += ("%s\t%s" % (mnemonic, op_str))+delimeter

        return assembly_code

    def getAssemblyCodeList(self,filename, bits='32bit'):

        iterable = self.diassemble(filename,bits)

        assembly_code_list = []
        for i in iterable:
            # To avoid TypeError: a bytes-like object is required, not 'str'
            (address, mnemonic, op_str) = i.address, i.mnemonic, i.op_str
            assembly_code_list.append("%s\t%s" % (mnemonic, op_str))

        return assembly_code_list


# fln = "/home/nislab2/Desktop/DissamblerEffect/benign/0b5511674394666e9d221f8681b2c2e6.exe"
# capstone = CapstoneDisasembler()
# print(capstone.getDisassembledCode(fln,bits="32bit"))
