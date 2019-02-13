import r2pipe
from dissamblerAbstract import disassemblerAbstract
from lib import fileutil as fu
import os


class Radare2Dissambler(disassemblerAbstract):

    def diassemble(self,filename, bits):
        r = r2pipe.open(filename)
        dirname,filename,cname = fu.fileNameSettings(filename,".asm","radare")

        return (r.cmd('pd $s >' + cname))

    def getDisassembledCode(self, filename, delimeter='\n', bits='32bit'):
        return self.diassemble(filename, bits)

    def getAssemblyCode(self, filename, delimeter='\n', bits='32bit'):
        return self.diassemble(filename, bits)


    def getAssemblyCodeList(self, filename, bits='32bit'):
        return self.diassemble(filename, bits)
