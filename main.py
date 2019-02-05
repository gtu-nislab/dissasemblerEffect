import distormDissasembler as distorm
import capstoneDissambler as cpstn
import ZydisDisasembler as zydis
#import pymsasidDissambler as pymsasid
import udis86Dissambler as udis
import radare2Dissambler as radare

import fileutil as fu
import os
import opcodeUtils

def writeFile(filename,data):
    with open(filename,"w") as f:
        f.write(data)

disasm = [
    #"capstone","distorm","zydis","udis",
          "radare"]

def asmbuilding(dir,disName,disassembler,isWriting=True):
    """
    extracting asm from files in dir by using belirtilen disassembler
    :param dir: directory where files are disassembled
    :param disName: disassembler name
    :param disassembler: abstract dissambler
    :return:
    """
    fls = fu.getFilePaths(dir,[".exe"])
    for fln in fls:
        dirname,filename,cname = fu.fileNameSettings(fln,".asm",disName)
        assembly =  disassembler.getDisassembledCode(fln)
        if isWriting:
            writeFile(cname,assembly)
        print("klasor:"+dirname+" dosyadi:"+filename+" disassembler:"+disName)


def opcodeExtraction(dir,delimeter=","):
    fls = fu.getFilePaths(dir,[".asm"])
    for fln in fls:
        dirname,filename,cname = fu.fileNameSettings(fln,".opcodeseq")
        a,d = opcodeUtils.get(fln)
        opcodeseq = opcodeUtils.opcodeSeq(a)
        print (cname)
        opss=delimeter.join(opcodeseq)
        writeFile(cname,opss)


def _main():

    dir= "/home/nislab2/Desktop/DissamblerEffect/"
    """
    for d in disasm:
        write = True
        if(d is "capstone"):
            disassembler = cpstn.CapstoneDisasembler()
        elif(d is "distorm"):
            disassembler = distorm.DistormDisassembler()
        elif(d is "zydis"):
            disassembler = zydis.ZydisDisassembler()
        elif(d is "radare"):
            disassembler = radare.Radare2Dissambler()
            write = False
        elif(d is "udis"):
            disassembler = udis.UdisDissambler()
        asmbuilding(dir,d,disassembler,write)
    """
    opcodeExtraction(dir)

_main()
