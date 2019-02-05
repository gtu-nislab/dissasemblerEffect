import subprocess
from dissamblerAbstract import disassemblerAbstract

#lib/ZydisDisasm -64 /home/nislab2/Desktop/DissamblerEffect/metamorphic/00fe0c08024f7db771d6711787d890a3.exe
class ZydisDisassembler(disassemblerAbstract):

    def diassemble(self,filename, bits='32bit'):
        """
            Disassembly executable file return iterable instruction set.

            :param filename : Executable file path
            :type filename: str
            :param bits : File platform 16, 32 or 64.
            :type bits : str [16bit, 32bit, 64bit] (default:32bit)
            :return: assembly code iterator:
            :rtype: str
        """
        mode = bits.replace("bit","")
        diasm = subprocess.check_output(['lib/ZydisDisasm',"-"+mode, filename])
        return diasm.decode("utf-8")

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


'''
zydisDissambler = ZydisDisasembler()
x = zydisDissambler.getDisassembledCode("/home/nislab2/Desktop/DissamblerEffect/metamorphic/00fe0c08024f7db771d6711787d890a3.exe")
print(x)
'''