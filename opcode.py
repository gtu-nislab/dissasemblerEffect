from dissambler import disassembler
import re
__disassemblers = ["zydis","capstone","IDA","distorm","Relyze"]

def getmnmonics():
    f ="/home/nislab2/Public/git/dissasemblerEffect/mnemonics.txt"
    mns = []
    with open(f,"r") as optxt:
        for l in optxt:
            mn = l.split("=")[0].strip()
            mns.append(mn)
    return mns
def createRegexSentence():
    s = "(?i)("#(mov|add|int|push|jb) "
    opcodelist = getmnmonics()
    for op in opcodelist:
        s+=op+"|"
    s =s[:-1] +") "
    return s
def extractopcodeFromline(line):
    #print(line)
    regstr = createRegexSentence()
    matchObj = re.search(regstr, line, re.M | re.I)
    s = line.split("       ")
    if (matchObj):
        s = matchObj.group(1)
        print(s)
        return s
    else:
        return False

def get(filename):
    global __disassemblers
    type=""
    for d in __disassemblers:
        if(d in filename):
            type=d
            break
    a=""
    with open(filename,encoding="utf8", errors='ignore') as df:
        a = df.read()

    return a,type

def opcodeSeq(assemblyCode):
    '''
    extract opcodes from assembly codes
    :param dissambler: type of disassembler
    :return:
    '''
    asscodelist = assemblyCode.split('\n')
    #print(asscodelist)
    opseq = []
    for ass in asscodelist:
        s = extractopcodeFromline(ass)
        if(s):
            opseq.append(s)
    return opseq

a,d=get("/home/nislab2/Desktop/DissamblerEffect/vxheaven_Relyze/3da90b51c040c16f662d2785b1931598.asm")
print(opcodeSeq(a))
