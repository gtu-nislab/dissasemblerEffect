import distormDissasembler as distorm
import capstoneDissambler as cpstn
import ZydisDisasembler as zydis

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

        capstone = cpstn.CapstoneDisasembler()
        distrm = distorm.DistormDisassemblerAbstract()
        zydisDissambler = zydis.ZydisDisassemblerAbstract()

        cname = dirname + "_capstone/" + filename + ".asm"
        disdir = dirname + "_distorm/" + filename + ".asm"
        zysdir = dirname + "_zydis/" + filename + ".asm"

        if not os.path.exists(dirname + "_capstone/"):
            os.makedirs(dirname + "_capstone/")
        if not os.path.exists(dirname + "_distorm/"):
            os.makedirs(dirname + "_distorm/")
        if not os.path.exists(zysdir + "_zydis/"):
            os.makedirs(zysdir + "_zydis/")
        writeFile(cname, capstone.getDisassembledCode(fln))
        writeFile(disdir, distrm.getDisassembledCode(fln))
        writeFile(zysdir, zydisDissambler.getDisassembledCode(fln))

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
