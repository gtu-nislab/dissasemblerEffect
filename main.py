import distormDissasembler as distorm
import capstoneDissambler as cpstn
import ZydisDisasembler as zydis
import pymsasidDissambler as pymsasid
import udis86Dissambler as udis

import fileutil as fu
import os
import opcodeUtils
def writeFile(filename,data):
    with open(filename,"w") as f:
        f.write(data)
def asmbuilding(dir):
    fls = fu.getFilePaths(dir,[".exe"])

    for fln in fls:
        dirname = os.path.dirname(fln)

        filename = os.path.basename(fln)
        (filename, ext) = os.path.splitext(filename)

#        capstone = cpstn.CapstoneDisasembler()
#        distrm = distorm.DistormDisassemblerAbstract()
#        zydisDissambler = zydis.ZydisDisassemblerAbstract()
        pymsasidDisasm = pymsasid.ZydisDisassemblerAbstract()
        udisDisasm = udis.ZydisDisassemblerAbstract()

#        cname = dirname + "_capstone/" + filename + ".asm"
#        disdir = dirname + "_distorm/" + filename + ".asm"
#        zysdir = dirname + "_zydis/" + filename + ".asm"

        udisdir = dirname + "udis/" + filename + ".asm"
        pymsasiddir = dirname + "pymsasid/" + filename + ".asm"
        """
        if not os.path.exists(dirname + "_capstone/"):
            os.makedirs(dirname + "_capstone/")
        if not os.path.exists(dirname + "_distorm/"):
            os.makedirs(dirname + "_distorm/")
        
        """
        if not os.path.exists(dirname + "_udisdir/"):
            os.makedirs(dirname + "_udisdir/")
        if not os.path.exists(dirname + "_pymsasiddir/"):
            os.makedirs(dirname + "_pymsasiddir/")
        """
        writeFile(cname, capstone.getDisassembledCode(fln))
        writeFile(disdir, distrm.getDisassembledCode(fln))
        writeFile(zysdir, zydisDissambler.getDisassembledCode(fln))
        """
        writeFile(udisdir, udisDisasm.getDisassembledCode(fln))
        writeFile(pymsasiddir, pymsasidDisasm.getDisassembledCode(fln))

def opcodeExtraction(dir,delimeter=","):
    fls = fu.getFilePaths(dir,[".asm"])
    for fln in fls:
        dirname = os.path.dirname(fln)
        filename = os.path.basename(fln)
        (filename, ext) = os.path.splitext(filename)
        a,d = opcodeUtils.get(fln)
        opcodeseq = opcodeUtils.opcodeSeq(a)
        ocsfn = dirname+os.sep+filename+".opcodeseq"
        print (ocsfn)
        opss=delimeter.join(opcodeseq)
        writeFile(ocsfn,opss)


def _main():

    dir= "/home/nislab2/Desktop/DissamblerEffect/"
    asmbuilding(dir)
    opcodeExtraction(dir)
_main()
