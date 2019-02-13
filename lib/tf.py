#!/usr/bin/env python
"""
    Calculate Term Frequency TF for given stream

#
# Author: Necmettin Çarkacı
#
# E-mail: necmettin [ . ] carkaci [ @ ] gmail [ . ] com
#
# Usage : tf.py file
#   dataset : file or directory
"""

from collections import Counter
import csv  # import csv library
from math import log10
def get_TF(filename, delimeter=',', normalization=''):
    """
    Calculate term frequency values given stream

    :param filename: The file which include stream
    :type filename:str
    :param delimeter: File content delimeter
    :type delimeter: str default value:','
    :param normalization: Normalization type. It can be total_sum, max, min, mean or default ''
    :type normalization: str default: ''
    :return:
    """

    with open(filename,'r') as input_file:
        term_content = Counter(input_file.read().split(delimeter))

    # Remove None object from dict
    term_content.pop('', None)
    normalizer = {}
    normalizer.update({ 'total_sum':sum(term_content.values()),
                        'max':max(term_content.values()),
                        'min':min(term_content.values()),
                        'mean':(sum(term_content.values())/len(term_content)),
                        '':1})

    tf_table = dict(map(lambda term: (term[0], term[1] / normalizer[normalization]),term_content.most_common() ))

    return tf_table
def csvbinary(csvfilepath,outcsvpath):
    '''
    reading histogram csv and producing binary csv
    :param csvfilepath: histogram csv filepath
    :param outcsvpath: binary csv filepath
    :return: dictionary list which all rows of binarycsv
    '''
    #HOWTO: read csv as dictlist, for each dict in dictlist; if val is null or 0 then 0, else 1
    with open(csvfilepath, 'r') as histcsv:
        print("csv file is reading:"+csvfilepath)
        dictlist = csv.DictReader(histcsv)
        fn_csv1 = set(dictlist.fieldnames)
        outdictlist = []
        with open(outcsvpath, "w") as binarycsv:
            writer = csv.DictWriter(binarycsv, delimiter=",", fieldnames=list(fn_csv1))
            writer.writeheader()
            for dict in dictlist:
                outdict = {}
                for v in dict.keys():
                    if dict[v]== "0" or dict[v]== '':
                        outdict[v]=0
                    else:
                        outdict[v]=1
                writer.writerow(outdict)
                outdictlist.append(outdict)
    return outdictlist



def csv_tfidf(csvfilepath,outcsvpath):
    '''
    reading histogram csv and producing tfidf csv
     formula : idf[x]: log(len(header)/header kaçtanesinde var) -> vektor
               tfidf[x][y] : tf[x][y] * idf[x] -> matrix

    :param csvfilepath: histogram csv filepath
    :param outcsvpath: tfidf csv filepath
    :return:
    '''
    with open(csvfilepath, 'r') as histcsv:
        print("csv file is reading:"+csvfilepath)
        dictlist = csv.DictReader(histcsv)
        dl = list(dictlist)
        headerlist = set(dictlist.fieldnames)
        print("csv file is reading:", len(headerlist))
        headerlist.remove("class_id")
        #headerlist.remove("")
        clids =[]
        for d in dl:
            clids.append(d.pop("class_id"))
            #d.pop("")
        headnumberx=len(dl)
        idf = {}
        outdictlist = []
        headerlength=len(headerlist)
        with open(outcsvpath, "w") as binarycsv:
            writer = csv.DictWriter(binarycsv, delimiter=",", fieldnames=(list(headerlist)+["class_id"]))
            writer.writeheader()
            for header in headerlist:
                #row row sayarken headercountperrow arttırılır
                if(header ==''):
                    continue
                headercountperrow=0
                #headercountperrow = sum([0 if row[header] is '' or row[header] is "0" else 1 for row in dl])
                for row in dl:
                    if not (row[header]== "0" or row[header]== ''):
                        headercountperrow +=1
                if headercountperrow==0:
                    idf[header]=0
                else:
                    idf[header] = (log10(headnumberx/(headercountperrow)))
            i=0
            for row in dl:
                r_sum = sum([int(v) if v.isdigit() else 0 for v in row.values()])  # tüm rowda toplam kaçtane var
                outdict = {}
               # if "class_id" in row.keys():
                 #   print("buldu")
                for k in row.keys():
                    if (k is ''):
                        continue
                    if (row[k].isdigit()):
                        h_row = int(row[k])
                    else:
                        h_row = 0 #'' demek, header rowda hiç geçmemiş
                    tfval = h_row / r_sum  # bu headerdakinin rowdaki toplam sayıya oranı term freq
                    outdict[k] = round((tfval * idf[k])*10000,3)

                outdict["class_id"]=clids[i]
                i+=1
                writer.writerow(outdict)
                outdictlist.append(outdict)
    return outdictlist

csv_tfidf("/media/nislab/5bece7d3-c708-491c-af60-c9e5278efd24/DATASET_SAMPLE_2000/opcodehistogram_csv/out_cleaned.csv","/media/nislab/5bece7d3-c708-491c-af60-c9e5278efd24/DATASET_SAMPLE_2000/opcodehistogram_csv/out_tfidf.csv")
#csvbinary("/media/nislab/5bece7d3-c708-491c-af60-c9e5278efd24/DATASET_SAMPLE/opcodehistogram_csv/out_100.csv","/media/nislab/5bece7d3-c708-491c-af60-c9e5278efd24/DATASET_SAMPLE/opcodehistogram_csv/out100_binary.csv")