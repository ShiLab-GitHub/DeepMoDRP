import csv
from smiles2graph import smile_to_graph
import pickle
from sklearn import preprocessing
import random
import numpy as np
from functions import TestbedDataset


def read_drug_list(filename):  # load drugs and their physicochemical properties from files
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    drug_dict = {}
    index = 0
    for line in reader:
        drug_dict[line[3]] = index  # build a dictionary to save the index of samples
        index += 1
    return drug_dict

def read_drug_smiles(filename, drug_dict):  # load drugs' SMILES
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    drug_smiles = [list() for i in range(len(drug_dict))]
    for line in reader:
        drug_smiles[drug_dict[line[3]]] = line[9]  # use the index in dictionary
    return drug_smiles


def read_cell_line_list(filename):  # load cell lines and build a dictionary
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    cell_line_dict = {}
    index = 0
    for line in reader:
        cell_line_dict[line[0]] = index
        index += 1
    return cell_line_dict


def read_cell_line_mut(filename, cell_line_dict):  # load one of the features of cell line - mut
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    mut = [list() for i in range(len(cell_line_dict))]
    for line in reader:
        if line[0] in cell_line_dict:
            mut[cell_line_dict[line[0]]] = line[1:]
    return mut

def read_cell_line_meth(filename, cell_line_dict):  # load one of the features of cell line - meth
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    meth = [list() for i in range(len(cell_line_dict))]
    for line in reader:
        if line[0] in cell_line_dict:
            meth[cell_line_dict[line[0]]] = line[1:]
    return meth


def min_max_nomalization(list, min, max):
    res = []
    for item in list:
        temp = (item - min) / (max - min)
        res.append(temp)
    return res


def get_all_graph(drug_smiles):
    smile_graph = {}
    for smile in drug_smiles:
        if len(smile) > 0:
            graph = smile_to_graph(smile)
            smile_graph[smile] = graph

    return smile_graph


def read_response_data_and_process(filename):
    # load features
    drug_dict= read_drug_list('data/drug/smile_inchi.csv')
    smile = read_drug_smiles('data/drug/smile_inchi.csv', drug_dict)
    smile_graph = get_all_graph(smile)
    cell_line_dict = read_cell_line_list('data/cellline/cellline_listwithACH_80cellline.csv')
    meth = read_cell_line_meth('data/cellline/METH_84cellline_378dim.csv', cell_line_dict)
    mut = read_cell_line_mut('data/cellline/MUT_85dim_2028dim.csv', cell_line_dict)
    copynumber = pickle.load(open('data/cellline/512dim_copynumber.pkl', 'rb'))  # Copy number pre-reduced by AE
    RNAseq = pickle.load(open('data/cellline/512dim_RNAseq.pkl', 'rb')) # GE pre-reduced by AE
    # feature normalization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    meth = min_max_scaler.fit_transform(meth)
    copynumber = min_max_scaler.fit_transform(copynumber)
    mut = min_max_scaler.fit_transform(mut)
    RNAseq = min_max_scaler.fit_transform(RNAseq)
    # read response data
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    data = []
    for line in reader:
        drug = line[0]
        cell_line = line[2]
        ic50 = float(line[7])
        data.append((drug, cell_line, ic50))
    random.shuffle(data)

    # match features and labels
    drug_smile = []
    cell_meth = []
    cell_copy = []
    cell_mut = []
    cell_RNAseq = []
    label = []
    for item in data:
        drug, cell_line, ic50 = item
        if drug in drug_dict and cell_line in cell_line_dict:
            drug_smile.append(smile[drug_dict[drug]])
            cell_meth.append(meth[cell_line_dict[cell_line]])
            cell_copy.append(copynumber[cell_line_dict[cell_line]])
            cell_mut.append(mut[cell_line_dict[cell_line]])
            cell_RNAseq.append(RNAseq[cell_line_dict[cell_line]])
            label.append(ic50)
    label = min_max_nomalization(label, min(label), max(label))

    # split data
    drug_smile = np.asarray(drug_smile)
    cell_meth, cell_copy, cell_mut, cell_RNAseq = np.asarray(cell_meth), np.asarray(cell_copy), np.asarray(cell_mut), np.asarray(cell_RNAseq)
    label = np.asarray(label)

    for i in range(5):
        total_size = drug_smile.shape[0]
        size_0 = int(total_size * 0.2 * i)
        size_1 = size_0 + int(total_size * 0.1)
        size_2 = int(total_size * 0.2 * (i + 1))
        # features of drug smiles
        drugsmile_test = drug_smile[size_0:size_1]
        drugsmile_val = drug_smile[size_1:size_2]
        drugsmile_train = np.concatenate((drug_smile[:size_0], drug_smile[size_2:]), axis=0)
        # features of cell meth
        cellmeth_test = cell_meth[size_0:size_1]
        cellmeth_val = cell_meth[size_1:size_2]
        cellmeth_train = np.concatenate((cell_meth[:size_0], cell_meth[size_2:]), axis=0)
        # features of cell copynumber
        cellcopy_test = cell_copy[size_0:size_1]
        cellcopy_val = cell_copy[size_1:size_2]
        cellcopy_train = np.concatenate((cell_copy[:size_0], cell_copy[size_2:]), axis=0)
        # features of cell mut
        cellmut_test = cell_mut[size_0:size_1]
        cellmut_val = cell_mut[size_1:size_2]
        cellmut_train = np.concatenate((cell_mut[:size_0], cell_mut[size_2:]), axis=0)
        # features of cell RNAseq
        cellRNAseq_test = cell_RNAseq[size_0:size_1]
        cellRNAseq_val = cell_RNAseq[size_1:size_2]
        cellRNAseq_train = np.concatenate((cell_RNAseq[:size_0], cell_RNAseq[size_2:]), axis=0)
        # label
        label_test = label[size_0:size_1]
        label_val = label[size_1:size_2]
        label_train = np.concatenate((label[:size_0], label[size_2:]), axis=0)

        TestbedDataset(root='data', dataset='train_set{num}'.format(num=i),
                       xds=drugsmile_train,
                       xcm=cellmeth_train, xcc=cellcopy_train, xcg=cellmut_train, xcr=cellRNAseq_train,
                       y=label_train, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='val_set{num}'.format(num=i),
                       xds=drugsmile_val,
                       xcm=cellmeth_val, xcc=cellcopy_val, xcg=cellmut_val, xcr=cellRNAseq_val,
                       y=label_val, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='test_set{num}'.format(num=i),
                       xds=drugsmile_test,
                       xcm=cellmeth_test, xcc=cellcopy_test, xcg=cellmut_test, xcr=cellRNAseq_test,
                       y=label_test, smile_graph=smile_graph)

    return


def process_blind_cell(filename):
    # load features
    drug_dict= read_drug_list('data/drug/smile_inchi.csv')
    smile = read_drug_smiles('data/drug/smile_inchi.csv', drug_dict)
    smile_graph = get_all_graph(smile)
    cell_line_dict = read_cell_line_list('data/cellline/cellline_listwithACH_80cellline.csv')
    meth = read_cell_line_meth('data/cellline/METH_84cellline_378dim.csv', cell_line_dict)
    mut = read_cell_line_mut('data/cellline/MUT_85dim_2028dim.csv', cell_line_dict)
    copynumber = pickle.load(open('data/cellline/512dim_copynumber.pkl', 'rb'))  # Copy number pre-reduced by AE
    RNAseq = pickle.load(open('data/cellline/512dim_RNAseq.pkl', 'rb')) # GE pre-reduced by AE
    # feature normalization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    meth = min_max_scaler.fit_transform(meth)
    copynumber = min_max_scaler.fit_transform(copynumber)
    mut = min_max_scaler.fit_transform(mut)
    RNAseq = min_max_scaler.fit_transform(RNAseq)
    # read response data
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    data = []
    for line in reader:
        drug = line[0]
        cell_line = line[2]
        ic50 = float(line[7])
        data.append((drug, cell_line, ic50))
    random.shuffle(data)

    dict_drug_cell = {}

    for item in data:
        drug, cell_line, ic50 = item
        if drug in drug_dict and cell_line in cell_line_dict:
            if cell_line in dict_drug_cell:
                dict_drug_cell[cell_line].append((drug, ic50))
            else:
                dict_drug_cell[cell_line] = [(drug, ic50)]

    for i in range(5):
        total_size = len(dict_drug_cell)
        size = int(total_size * i * 0.2)
        size1 = size + int(total_size * 0.1)
        size2 = int(total_size * (i + 1) * 0.2)

        drugsmile_train = []
        cellmeth_train = []
        cellcopy_train = []
        cellmut_train = []
        cellRNAseq_train = []
        label_train = []

        drugsmile_val = []
        cellmeth_val = []
        cellcopy_val = []
        cellmut_val = []
        cellRNAseq_val = []
        label_val = []

        drugsmile_test = []
        cellmeth_test = []
        cellcopy_test = []
        cellmut_test = []
        cellRNAseq_test = []
        label_test = []

        pos = 0
        for cell, values in dict_drug_cell.items():
            pos += 1
            for v in values:
                drug, ic50 = v
                if pos >= size and pos < size1:
                    drugsmile_test.append(smile[drug_dict[drug]])
                    cellmeth_test.append(meth[cell_line_dict[cell]])
                    cellcopy_test.append(copynumber[cell_line_dict[cell]])
                    cellmut_test.append(mut[cell_line_dict[cell]])
                    cellRNAseq_test.append(RNAseq[cell_line_dict[cell]])
                    label_test.append(ic50)
                elif pos >= size1 and pos < size2:
                    drugsmile_val.append(smile[drug_dict[drug]])
                    cellmeth_val.append(meth[cell_line_dict[cell]])
                    cellcopy_val.append(copynumber[cell_line_dict[cell]])
                    cellmut_val.append(mut[cell_line_dict[cell]])
                    cellRNAseq_val.append(RNAseq[cell_line_dict[cell]])
                    label_val.append(ic50)
                else:
                    drugsmile_train.append(smile[drug_dict[drug]])
                    cellmeth_train.append(meth[cell_line_dict[cell]])
                    cellcopy_train.append(copynumber[cell_line_dict[cell]])
                    cellmut_train.append(mut[cell_line_dict[cell]])
                    cellRNAseq_train.append(RNAseq[cell_line_dict[cell]])
                    label_train.append(ic50)

        drugsmile_train= np.asarray(drugsmile_train)
        cellmeth_train, cellcopy_train, cellmut_train, cellRNAseq_train = np.asarray(cellmeth_train), np.asarray(cellcopy_train), np.asarray(cellmut_train), np.asarray(cellRNAseq_train)
        label_train = np.asarray(label_train)

        drugsmile_val = np.asarray(drugsmile_val)
        cellmeth_val, cellcopy_val, cellmut_val, cellRNAseq_val = np.asarray(cellmeth_val), np.asarray(cellcopy_val), np.asarray(cellmut_val), np.asarray(cellRNAseq_val)
        label_val = np.asarray(label_val)

        drugsmile_test = np.asarray(drugsmile_test)
        cellmeth_test, cellcopy_test, cellmut_test, cellRNAseq_test = np.asarray(cellmeth_test), np.asarray(cellcopy_test), np.asarray(cellmut_test), np.asarray(cellRNAseq_test)
        label_test = np.asarray(label_test)

        TestbedDataset(root='data', dataset='train_blind_cell{num}'.format(num=i),
                       xds=drugsmile_train,
                       xcm=cellmeth_train, xcc=cellcopy_train, xcg=cellmut_train, xcr=cellRNAseq_train,
                       y=label_train, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='val_blind_cell{num}'.format(num=i),
                       xds=drugsmile_val,
                       xcm=cellmeth_val, xcc=cellcopy_val, xcg=cellmut_val, xcr=cellRNAseq_val,
                       y=label_val, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='test_blind_cell{num}'.format(num=i),
                       xds=drugsmile_test,
                       xcm=cellmeth_test, xcc=cellcopy_test, xcg=cellmut_test, xcr=cellRNAseq_test,
                       y=label_test, smile_graph=smile_graph)

    return


def process_blind_drug(filename):
    # load features
    drug_dict= read_drug_list('data/drug/smile_inchi.csv')
    smile = read_drug_smiles('data/drug/smile_inchi.csv', drug_dict)
    smile_graph = get_all_graph(smile)
    cell_line_dict = read_cell_line_list('data/cellline/cellline_listwithACH_80cellline.csv')
    meth = read_cell_line_meth('data/cellline/METH_84cellline_378dim.csv', cell_line_dict)
    mut = read_cell_line_mut('data/cellline/MUT_85dim_2028dim.csv', cell_line_dict)
    copynumber = pickle.load(open('data/cellline/512dim_copynumber.pkl', 'rb'))  # Copy number pre-reduced by AE
    RNAseq = pickle.load(open('data/cellline/512dim_RNAseq.pkl', 'rb')) # GE pre-reduced by AE
    # feature normalization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
    meth = min_max_scaler.fit_transform(meth)
    copynumber = min_max_scaler.fit_transform(copynumber)
    mut = min_max_scaler.fit_transform(mut)
    RNAseq = min_max_scaler.fit_transform(RNAseq)
    # read response data
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    data = []
    for line in reader:
        drug = line[0]
        cell_line = line[2]
        ic50 = float(line[7])
        data.append((drug, cell_line, ic50))
    random.shuffle(data)

    dict_drug_cell = {}

    for item in data:
        drug, cell_line, ic50 = item
        if drug in drug_dict and cell_line in cell_line_dict:
            if drug in dict_drug_cell:
                dict_drug_cell[drug].append((cell_line, ic50))
            else:
                dict_drug_cell[drug] = [(cell_line, ic50)]

    for i in range(5):
        total_size = len(dict_drug_cell)
        size = int(total_size * i * 0.2)
        size1 = size + int(total_size * 0.1)
        size2 = int(total_size * (i + 1) * 0.2)

        drugsmile_train = []
        cellmeth_train = []
        cellcopy_train = []
        cellmut_train = []
        cellRNAseq_train = []
        label_train = []

        drugsmile_val = []
        cellmeth_val = []
        cellcopy_val = []
        cellmut_val = []
        cellRNAseq_val = []
        label_val = []

        drugsmile_test = []
        cellmeth_test = []
        cellcopy_test = []
        cellmut_test = []
        cellRNAseq_test = []
        label_test = []

        pos = 0
        for drug, values in dict_drug_cell.items():
            pos += 1
            for v in values:
                cell, ic50 = v
                if pos >= size and pos < size1:
                    drugsmile_test.append(smile[drug_dict[drug]])
                    cellmeth_test.append(meth[cell_line_dict[cell]])
                    cellcopy_test.append(copynumber[cell_line_dict[cell]])
                    cellmut_test.append(mut[cell_line_dict[cell]])
                    cellRNAseq_test.append(RNAseq[cell_line_dict[cell]])
                    label_test.append(ic50)
                elif pos >= size1 and pos < size2:
                    drugsmile_val.append(smile[drug_dict[drug]])
                    cellmeth_val.append(meth[cell_line_dict[cell]])
                    cellcopy_val.append(copynumber[cell_line_dict[cell]])
                    cellmut_val.append(mut[cell_line_dict[cell]])
                    cellRNAseq_val.append(RNAseq[cell_line_dict[cell]])
                    label_val.append(ic50)
                else:
                    drugsmile_train.append(smile[drug_dict[drug]])
                    cellmeth_train.append(meth[cell_line_dict[cell]])
                    cellcopy_train.append(copynumber[cell_line_dict[cell]])
                    cellmut_train.append(mut[cell_line_dict[cell]])
                    cellRNAseq_train.append(RNAseq[cell_line_dict[cell]])
                    label_train.append(ic50)

        drugsmile_train = np.asarray(drugsmile_train)
        cellmeth_train, cellcopy_train, cellmut_train, cellRNAseq_train = np.asarray(cellmeth_train), np.asarray(cellcopy_train), np.asarray(cellmut_train), np.asarray(cellRNAseq_train)
        label_train = np.asarray(label_train)

        drugsmile_val = np.asarray(drugsmile_val)
        cellmeth_val, cellcopy_val, cellmut_val, cellRNAseq_val = np.asarray(cellmeth_val), np.asarray(cellcopy_val), np.asarray(cellmut_val), np.asarray(cellRNAseq_val)
        label_val = np.asarray(label_val)

        drugsmile_test = np.asarray(drugsmile_test)
        cellmeth_test, cellcopy_test, cellmut_test, cellRNAseq_test = np.asarray(cellmeth_test), np.asarray(cellcopy_test), np.asarray(cellmut_test), np.asarray(cellRNAseq_test)
        label_test = np.asarray(label_test)

        TestbedDataset(root='data', dataset='train_blind_drug{num}'.format(num=i),
                       xds=drugsmile_train,
                       xcm=cellmeth_train, xcc=cellcopy_train, xcg=cellmut_train, xcr=cellRNAseq_train,
                       y=label_train, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='val_blind_drug{num}'.format(num=i),
                       xds=drugsmile_val,
                       xcm=cellmeth_val, xcc=cellcopy_val, xcg=cellmut_val, xcr=cellRNAseq_val,
                       y=label_val, smile_graph=smile_graph)
        TestbedDataset(root='data', dataset='test_blind_drug{num}'.format(num=i),
                       xds=drugsmile_test,
                       xcm=cellmeth_test, xcc=cellcopy_test, xcg=cellmut_test, xcr=cellRNAseq_test,
                       y=label_test, smile_graph=smile_graph)

    return


if __name__ == "__main__":
    read_response_data_and_process('data/ic50/80cell_line_ic50.csv')
    #process_blind_cell('data/ic50/80cell_line_ic50.csv')
    #process_blind_drug('data/ic50/80cell_line_ic50.csv')
