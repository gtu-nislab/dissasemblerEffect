import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from lib import fileutil as fu


#3-5 ml algoritmasi implement edilir. sonuclari cikarilir. sonuclar grafiklendirilir, sonuclar ayni tablo ustunde karsilastirilir,
# zaman tutulmali
def ml_timer(ml_func):
    import time
    start = time.time()
    r =ml_func()
    end = time.time()
    return end - start, r
def hid_mark_modeling(sequences):
    from hmmlearn import hmm
    model = hmm.GaussianHMM(n_components=5, n_iter=10)
    model.fit(sequences)
    return model #model.score(seq) ile score hesaplanir, belirlenen threshold ile score degeri ile tespit edilir
def rand_forest(X,y,Xtst):
    from sklearn.ensemble import RandomForestClassifier
    print(X.shape)
    print(len(set(y)))
    clf = RandomForestClassifier(n_estimators=100,random_state=0)
    clf.fit(X, y)
    print(clf.feature_importances_)
    y_pre = clf.predict(Xtst)
    return y_pre
def supp_vec_mac(X,y,Xtst):
    from sklearn import svm
    print(X.shape)
    print(len(set(y)))
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)
    y_pre = clf.predict(Xtst)
    return y_pre

def k_nearest_neigh(n_neighbors,X,y,Xtst):
    from sklearn.neighbors import KNeighborsClassifier
    print(X.shape)
    print(len(set(y)))
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X, y)
    y_pre = neigh.predict(Xtst)
    return y_pre

def naiv_bayes(X,y,Xtst):
    from sklearn.naive_bayes import GaussianNB
    print(X.shape)
    print(len(set(y)))
    clf = GaussianNB()
    clf.fit(X, y)
    y_pre = clf.predict(Xtst)
    return y_pre

def multi_layer_perceptron(X,y,Xtst,optimizer="adam",activation="relu", hidden_layer_size=(20,2)): #solver weight optimization,
    from sklearn.neural_network import MLPClassifier
    print(X.shape)
    print(len(set(y)))
    clf = MLPClassifier(solver=optimizer,activation=activation, hidden_layer_sizes=hidden_layer_size)
    clf.fit(X, y)
    y_pre = clf.predict(Xtst)
    return y_pre

#evaluate
def cal_acc(y_actual,y_pred):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_actual, y_pred)

def cal_recall(y_actual,y_pred):
    from sklearn.metrics import recall_score
    return recall_score(y_actual, y_pred)

def cal_precision(y_actual,y_pred):
    from sklearn.metrics import precision_score
    return precision_score(y_actual, y_pred)

def cal_f1(y_actual,y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_actual, y_pred)

def roc_curve(y_true,p_probas,fig):
    import scikitplot as skplt
    import matplotlib.pyplot as plt
    i=0
    y_probas = []
    for p in p_probas:
        y_probas.append(p[0][y_true[i]])
    skplt.metrics.plot_roc_curve(y_true, y_probas)
    plt.savefig(fig)


#graphics - Colored and Styled Bar Chart for opcode counts, basarimlarin karsilastirmasi-Specify Binning Function histogram, Group By in Python for each disasm, (roc_curve),
#sequential icin nasil bir gosterim?

figureCount=0
color = ['black', 'red', 'green', 'blue', 'cyan', "yellow"]

def onebarchart(y,x,chartfig):
    y_pos = np.arange(len(y))
    global figureCount
    plt.figure(figureCount)
    figureCount += 1
    plt.bar(y_pos, x, color=color)
    # use the plt.xticks function to custom labels
    plt.xticks(y_pos, y, color='black', rotation=45, fontsize='7', horizontalalignment='right')
    # remove labels
    # plt.tick_params(labelbottom='off')
    plt.savefig(chartfig)
def multiplebarchart(y_on_legend,x,multi,chartfig):
    y_pos = np.arange(1, 2 * len(multi) + 1,
                      2)  # key sayisi kadar opsample var oyuzden okadar tane cubuk olacak
    global figureCount
    plt.figure(figureCount)
    figureCount += 1
    i = 0
    for y_val in y_on_legend:
        # her opcode turu icin 6 disAsm icinde bir gosterim olur ve her disAsm farkli renk
        data = x[i]  # her disAssembler icin farkli data farkli renk
        plt.bar(y_pos + (i * 0.1 - 0.2), data, width=0.1, color=color[i])
        i += 1
    y_key = multi
    plt.xticks(y_pos, y_key, color='black', rotation=45, fontsize='7', horizontalalignment='right')
    plt.legend(y_on_legend)
    plt.savefig(chartfig)
def chart_for_opfreq(disAsms,opcodes,chartfig):
    """
    showing opcode count which are assembled by different disAsms on histogram
    :param disAsms: disassembler list
    :param opcodes: opcode count list- if list of number, total opcode for each disassembler, if list of dict, some selected opcode count for each disassembler oc=[{add:12,move:21},{add:13,move:22},...]
    :return:
    """
    if(isinstance(opcodes[0],int)):
        onebarchart(disAsms,opcodes,chartfig)
    elif(isinstance(opcodes[0],dict)):
        x = [ freq.values() for freq in opcodes]
        ops = opcodes[0].keys()
        multiplebarchart(disAsms,x,ops,chartfig)
    else:
        raise Exception("it should be dictionary or integer: IT CAN NOT BE SHOWN ON CHART, ERROR WHILE OPCODE FREQUENCY SHOWN")

def chart_for_files(disAsms,x,y,chartfig):
    """
    each samples of file are dot on chart. tsne is applied to show file. for each disassembler, there is a different colpur and for each classes there is a different marker
    :param disAsms: list of disassemblers
    :param x: data - (d,s,f) where d is number of disassembler, s is number of sample, f is number of feature
    :param y: class - (d,s) where d is number of disassembler, s is number of sample, values are taken from set of classes e.g. if 6 classes then values can be 0,1,2,3,4,5.
    :return:
    """
    import tsne
    global figureCount
    plt.figure(figureCount)
    figureCount += 1
    i=0
    marker = [(4, 0),(4, 2),(5, 3),(5, 1),(4, 2),(3,1),(2,1)]
    m = [marker[y_i] for y_i in y]
    for d in disAsms:
        print(d+" degerleri icin tsne uygulaniyor")
        Xn = np.array(x[i]).astype(np.float32)
        Y = tsne.tsne(Xn, 2, 50, 20.0)
        print("tsne bitti simdi grafige yazdiriliyor renk:"+color[i])
        plt.scatter(Y[:, 0], Y[:, 1], s=80, marker=m, c=color[i])
        i+=1
    plt.legend(disAsms)
    plt.savefig(chartfig)

def chart_for_success(disAsms,success,chartfig):
    """
    showing success value which are gotten from different assembler on histogram
    :param disAsms: disassembler list
    :param success: success valuet list- if list of number, acc of randomForest for each disassembler, if list of dict, acc taken from different ML methods  for each disassembler suc = [{rf:12,knn:21},{rf:13,knn:22},...]
    :param chartfig:
    :return:
    """
    if(isinstance(success[0],int)):
        onebarchart(disAsms,success,chartfig)
    elif(isinstance(success[0],dict)):
        x = [ acc.values() for acc in success]
        mls = success[0].keys()
        multiplebarchart(disAsms,x,mls,chartfig)
    else:
        raise Exception("it should be dictionary or integer: IT CAN NOT BE SHOWN ON CHART, ERROR WHILE OPCODE FREQUENCY SHOWN")

def csv2numpy(filename):
    from numpy import genfromtxt
    my_data = genfromtxt(filename, delimiter=',',dtype=None,encoding=None)
    feature=my_data[0]
    data = my_data[1:]
    class_index = np.where(feature == 'class_id')[0][0]
    hash_index = np.where(feature == 'hash')[0][0]
    y =  data[: , class_index]
    hashes = data[: , hash_index]
    x = np.delete(data,[class_index,hash_index],axis=1)
    x[x =='']=0
    return x.astype(int),y,hashes,feature #return x,y,hashes,features

def clean(bigclasname):
    ccc = bigclasname.split("/")
    return ccc[-1]

def crossvalid(x,y,h,feature,cv=2):
    """
    prepare data for cross validation
    :param x: data
    :param y: labels
    :param h: sample id's - hashes
    :param feature:
    :param cv: cross validation
    :return: xtr, xts, ytr, yts, htr, hts
    """

    shape  = x.shape
    s = shape[0] # sample count
    f = len(feature)-2 # feature count
    sr = int(s / cv) # sample count in one set

    #define matrices
    xts = np.zeros(shape=(cv, sr, f))
    yts = np.chararray(shape=(cv, sr),itemsize=64)
    hts = np.chararray(shape=(cv, sr),itemsize=32)

    xtr = np.zeros(shape=(cv, s-sr, f))
    ytr = np.chararray(shape=(cv, s-sr),itemsize=64)
    htr = np.chararray(shape=(cv, s-sr),itemsize=32)
    for i in range(1,cv+1):
        xts[i-1] = x[(i-1)*sr:i*sr]
        yts[i-1] = np.vectorize(clean)(y[(i-1)*sr:i*sr])
        hts[i-1] = h[(i-1)*sr:i*sr]
        if i-1 == 0:
            xtr[i-1] = x[i*sr:]
            ytr[i-1] = np.vectorize(clean)(y[i*sr:])
            htr[i-1] = h[i*sr:]
        elif i == cv:
            xtr[i-1] = x[0:(i-1)*sr]
            ytr[i-1] = np.vectorize(clean)(y[0:(i-1)*sr])
            htr[i-1] = h[0:(i-1)*sr]

        else:
            xtr[i-1] = np.concatenate((x[0:(i-1)*sr] , x[i*sr:]))
            ytr[i-1] = np.vectorize(clean)(np.concatenate((y[0:(i-1)*sr] , y[i*sr:])))
            htr[i-1] = np.concatenate((h[0:(i-1)*sr] , h[i*sr:]))
    return xtr, xts, ytr, yts, htr, hts

def make(dirname,type="c", cv=10):
    '''
    train set test set and their class values are prepared according to cross-validation and tag type.
    :param type: flag for classes of data. In (d)etection mode, there are 2 classes, malware and benign. In classicifation mode, all different tags are class
    :param cv: cross validation number
    :return: according to cross validation, 4 matrix; xtrain (cv,sr,f), xtest(cv,ss,f), ytrain(cv,sr), ytest(cv,ss) - cv: cross-valid, sr:count of sample in train set, f:feature number,
    ss:count of sample in test set
    '''
    csvfiles = fu.getFilePaths(dirname,extensionList=[".csv"])

    if type is "c":
        for cfile in csvfiles:
            data= csv2numpy(cfile)
            shp = data.shape
            sr = shp[0]
            f = shp[1]
            xtr = np.zeros(shape=(cv, sr, f))

        print("prepare dataset for classification")
    elif type is "d":
        print("prepare dataset for detection")
    else:
        raise Exception("MAKE DATASET ERROR!!! NO PERMISSION FOR MAKE FOR ANOTHER TYPE")

if __name__ is "__mainx__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
    print(X.shape)
    print(len(set(y)))
    Xtest = np.random.rand(30,4)
    #timer and rand_forest testing
    def ml_func():
        return rand_forest(X, y, Xtest)
    t,y_pre = ml_timer(ml_func)
    print("time:"+str(t))
    y_tst =np.random.randint(0,2,30)
    #accuracy testing
    print("y_tst:"+str(y_tst)+"y_pre:"+str(y_pre))
    acc = cal_acc(y_tst,y_pre)
    print("acc for random_forest:"+str(acc))

x, y, hashes, feature =csv2numpy("/home/nislab2/Desktop/DissamblerEffect/metamorphic_zydis/csv/opcode_histogram.csv")
#data = np.arange(1120).reshape(data.shape)
print(str(crossvalid(x,y,hashes,feature,cv=5)))