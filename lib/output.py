

import os
import csv  # import csv library
import sqlite3
from collections import OrderedDict
from lib import fileutil
import os, sys, time
import re
def writeIntoFile(filename, extension, content, sep=','):

    # start function variables
    dirname         = os.path.dirname(filename)


    filename        = os.path.basename(filename)
    (filename, ext) = os.path.splitext(filename)
    output_filename = dirname+os.sep+extension+os.sep+filename+'.'+extension

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as output_file:
        output_file.write(content)

def getAttrList(dictlist):
    # dictlist den fieldnameleri topla (ilk deger hash)
    fset = set()
    fs_hash = set()

    for dictionary in dictlist:
        
        dk = list(dictionary.keys())
        if dk.__contains__("hash"):
            dk.remove("hash")
        fset.update(dk)
        fs_hash.add(dictionary["hash"])  # hashleri topla

    fs_hash.update()
    return fset,fs_hash
def writeSingleIntoCSVFile(filename,dictlist, delimeter, outfile=''):
    print("features are collecting")
    fset, fs_hash = getAttrList(dictlist) #fset is all fields in dictionary

    fset.add("hash")
    print("done: "+str(len(fset)))
    csv_filename = filename + '.csv'
    directory = os.path.dirname(csv_filename)
    print(csv_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(csv_filename, 'w') as output_file:

        writer = csv.DictWriter(output_file,delimiter=delimeter,fieldnames=list(fset))
        writer.writeheader()
        t = len(dictlist)
        i=1
        for d in dictlist:#adding fields which is not in d but in other, to d
            len_d = len(fset.difference(set(d.keys())))
            if(len_d >=0 ):
                print("length of difference: "+str(len_d))
                #for k  in fset.difference(set(d.keys())):
                #   d[k]=0
            print(str(i)+" - "+str(t))
            i+=1
            writer.writerow(d)
        print("csv has just been written")

def writeIntoCSVFile(filename,dictlist, delimeter=','):
    """
           alg: dictlist den fieldnameleri topla (ilk deger hash), fieldnameler colon adi, ayni hash olan dict'lerin valuelerini keylerine göre yerleştir.
       """

    fset,fs_hash = getAttrList(dictlist)

    flist = list(fset)
    print(len(flist))
    csv_filename = filename + '.csv'
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    with open(csv_filename, 'w') as output_file:
        writer = csv.writer(output_file,delimiter=delimeter)
        flist.remove("class_id")
        flist =  flist+["class_id"]
        writer.writerow(["hash"]+flist) #basliklar yazılıyor - ilki hash degeri

        #her bir hash degeri icin yeni row olustur -> h,fset[0],fset[1],... ->
        for h in fs_hash:
            row = [h]
            for dictionary in (dl for dl in dictlist if h in dl.values()):#hash degeri olan dictler
                i = 0
                for fieldname in flist: #fset ile header row u atandı, oyüzden buradaki sıraya göre yazılmalı
                    if fieldname in dictionary.keys():
                        if (len(row)>len(fset)):
                            try:
                                row[flist.index(fieldname)+1]=str(dictionary[fieldname]).replace(delimeter,'')
                            except:
                                print("hatanın koptugu yer: "+fieldname+str(flist.index(fieldname)+1)+str(i))
                        else:
                            i+=1
                            row.append(str(dictionary[fieldname]).replace(delimeter,' '))
                    else:
                        if (len(row)>len(fset)): #taştı
                            continue
                        row.append("0")
            row = [0 if v is "" or v is " " else v for v in row]
            #classid sona alma
            row = classidsona(row)
            writer.writerow(row)

def isStr(a):
    if type(a) is str:
        return True
    else:
        return False


def classidsona(row):
    r = re.compile("\/media\/nislab\/")

    filterresult = filter(r.match, filter(isStr, row))
    clid=""
    clnew =""
    for cl in filterresult:
        if "BENIGN" in cl:
            clnew =cl.replace(cl.split("BENIGN",1)[1],"")
        clid = cl

    if not clid is "":
        row.remove(clid)
        if clnew is not "":
            row.append(clnew)
        else:
            row.append(clid)
    return row
def mergeCSV(csv1,csv2,csvOut):
    f1= open(csv1)
    reader1 = csv.DictReader(f1)
    fn_csv1= set(reader1.fieldnames)
    f2= open(csv2)
    reader2 = csv.DictReader(f2)
    dictlist = list(reader1)+list(reader2)
    f1.close()
    f2.close()
    return writeIntoCSVFile(csvOut,dictlist)

def mergeAll(filespath,csvOut,selectedColumn=[]):
    files = fileutil.getFilePaths(filespath)
    dictlist = []
    for file in files:
        f1 = open(file)
        reader1 = csv.DictReader(f1)
        dicts =(list(reader1))
        if(selectedColumn is not []):
            dicts = narrowDict(dicts,selectedColumn)
        dictlist +=dicts
        f1.close()
    return writeIntoCSVFile(csvOut, dictlist)

def narrowDict(dicts, selectedColumns):
    sc = list(set(list(dicts[0].keys())).intersection(set(selectedColumns))) #selected columns in this dictionary has just found.
    rds = []
    for d in dicts:
        rds.append({i:d[i] for i in d if i in sc})
    return rds

def writeFeaturesIntoDB(dbName, dictlist ):
    conn = sqlite3.connect(dbName)
    cursor = conn.cursor()
    fset,fs_hash = getAttrList(dictlist)

    # Create table
    cursor.execute('CREATE TABLE IF NOT EXISTS ' + 'features' + ' ' + list(fset))
    #inaert->
    # Save (commit) the changes
    conn.commit()

    cursor.execute()
    # We can also close the connection if we are done with it.
    # Just be sure any changes have been committed or they will be lost.
    conn.close()

def writeFeatureAsSequence(filename, extension, sequence, sep=',',outfile=''):

    # start function variables
    if(outfile==''):
        dirname         = os.path.dirname(filename)
    else:
        dirname         = outfile

    filename        = os.path.basename(filename)
    (filename, ext) = os.path.splitext(filename)
    output_filename = dirname+os.sep+extension+os.sep+filename+'.'+extension

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'w') as output_file:
        for s in sequence:
            output_file.write(str(s)+sep)

if __name__ == '__main__':

    f1 = 'data/metamorphic.csv'
    f2 = 'data/bening.csv'
    f3 = 'data/out.csv'

    #mergeCSV(f1, f2, f3)
   # mergeAll("/home/nislab3/Desktop/outs/dynamic_apistat/APISTATS_100/dataset1_4class/files/",'/home/nislab3/Desktop/outs/dynamic_apistat/APISTATS_100/dataset1_4class/apistat_out')
    mergeAll("/home/nislab3/Desktop/anomalies/other",   '/home/nislab3/Desktop/anomalies/other/out')

"""
if __name__ == '__main__':
    #directory = "samples"
    directory = sys.argv[1]

    start = time.time()
    print("Start time : " + str(time.clock()))
    extensionList = ['.exe', '.dll']

    listOfFile = fileutil.getFilePaths(directory, extensionList)
    dictlist1 = []

    for filename in listOfFile:
        opcodes = OpCode.getOpcodeFrequency(filename)
        #generalInfos = generalInfo.getGenericFileInformation(filename)
        #generalInfos_dict=OrderedDict(dict(generalInfos))
        opcodes_dict=OrderedDict(dict(opcodes))

        #generalInfos_dict["hash"] = filename
        opcodes_dict["hash"] = filename

        #print(generalInfos)
        #print(generalInfos_dict)
        #dictlist1 +=[generalInfos_dict]
        dictlist1 +=[opcodes_dict]

    #print (dictlist1)
    print(writeIntoCSVFile("featuretable",dictlist1))
    end = time.time()
    print('End time : ' + str(time.clock()))
    print('Running time : ' + str(end - start))
"""
