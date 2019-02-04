# test1.py
import octopus
from dissamblerAbstract import disassemblerAbstract

class octobusDissambler(disassemblerAbstract):
    def diassemble(self, filename, bits):
        pass
    def getDisassembledCode(self, filename, delimeter='\n', bits='32bit'):
        pass

    def getAssemblyCode(self, filename, delimeter='\n', bits='32bit'):
        pass

    def getAssemblyCodeList(self, filename, bits='32bit'):
        pass