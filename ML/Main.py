import os
import histogramCalculater
from lib import fileutil, output

def core_process(process_function, output_ext, delimeter=',', dataset='dataset', extensionList=['.exe', '.dll'],feature_type="text"):
    """
    Get a dataset collect all information and  write them into file.

    Attention ! Assume that filename is file hash value or unique value for the file

    :param data_name: Dataset name
    :type data_name:str
    :param extensionList: File extension list
    :type extensionList: list
    :return : void
    """

    if os.path.isfile(dataset):
        pass
    elif os.path.isdir(dataset):
        listOfFile = fileutil.getFilePaths(dataset, extensionList)

        # Collect information from source file
        for index, filename in enumerate(listOfFile):

            # Get content
            content = process_function(filename, delimeter=delimeter)
            if content is not None:
                try:
                    if feature_type =="seq":
                        output.writeFeatureAsSequence(filename, output_ext, content)
                    else:
                        output.writeIntoFile(filename, output_ext, content)
                except IOError as ioe:
                    print(str(ioe))
                    pause()
            print(str(index)+"  -  " +str(len(listOfFile)))
    else:
        print('File type must be file or directory.')


def feature_extraction(process_function, outputfilename, delimeter=',', dataset='dataset',
                       extensionList=['.exe', '.dll']):
    """
    Get a dataset collect all information and  write them into file.

    Attention ! Assume that filename is file hash value or unique value for the file
    :param process_function: the function pointer which will be executed to extract feature. Key point here is that its return type is dictionary
    :type process_function:func*
    :param outputfilename: name of documentary which the result of csv file will be put
    :type outputfilename: str
    :param dataset: Dataset name
    :type dataset:str
    :param extensionList: File extension list whose is on processing
    :type extensionList: list

    :return : void
    """
 
    if os.path.isfile(dataset):
        pass
    elif os.path.isdir(dataset):
        listOfFile = fileutil.getFilePaths(dataset, extensionList)

        # Collect information from source file
        content_list = []
        for index, filename in enumerate(listOfFile):
            # Assume that filename is file hash value
            hash_id  = os.path.basename(filename).split('.')[0]
            class_id = os.path.dirname(filename)

            # Get content
            #filename = os.getcwd()+os.sep+filename

            content = process_function(filename, delimeter=delimeter)

            if content is not None:
                content['hash']     = hash_id
                content['class_id'] = class_id
                content_list.append(content)

            print(outputfilename+"-"+str(index)+"  -  "+ str(len(listOfFile)))
 
            # Write informations into csv file
            outfile = os.path.dirname(filename)+os.sep+'csv'+os.sep+outputfilename
            try:
                output.writeSingleIntoCSVFile( outfile , content_list, delimeter)
            except IOError as ioe:
                print(str(ioe))
                pause()

    else:
        print('File type must be file or directory.')

def pause():
    wait = input("PRESS ENTER TO CONTINUE.")

#def seq_out()

if __name__ == '__main__':
    datasets = [
        #"dataset"
        "/home/nislab2/Desktop/DissamblerEffect/metamorphic_zydis",
        #"/home/nislab2/Desktop/DATASET/DATASET_SAMPLE 100/METAMORPHIC/MWOR",
        #"/home/nislab2/Desktop/DATASET/DATASET_SAMPLE 100/METAMORPHIC/PSMPC",
        #"/media/nislab/5bece7d3-c708-491c-af60-c9e5278efd24/DATASET_SAMPLE_1000/GENERAL_MAL/VXHEAVEN"
        #"/media/nislab/5bece7d3-c708-491c-af60-c9e5278efd24/DATASET_SAMPLE_1000/METAMORPHIC"
    ]

    #dataset='/media/nislab/5bece7d3-c708-491c-af60-c9e5278efd24/DATASET_ALL/METAMORPHIC/NGVCK' #/media/nislab/5bece7d3-c708-491c-af60-c9e5278efd24/DATASET_ALL/METAMORPHIC/G2
    for dataset in datasets :
        extensionList = [".exe",".dll"]


        feature_extraction(histogramCalculater.getHistogram,            'opcode_histogram',             extensionList=['.opcode'],dataset=dataset)
        #feature_extraction(histogramCalculater.getHistogram,            'opcode_bigram_histogram',      extensionList=['.opcodebigram'],dataset=dataset)
