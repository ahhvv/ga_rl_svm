import pandas as pd

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


Dos_type = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb', 'processtable', 'udpstorm', 'apache2', ]
Probe_type = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
R2L_type = ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'xlock', 'xsnoop', 'snmpguess',
            'snmpgetattack', 'sendmail', 'named', 'worm', 'warezclient']
U2R_type = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps', 'httptunnel']


def type_sort(x):
    if x in Dos_type:
        return 0
    elif x in Probe_type:
        return 0
    elif x in R2L_type:
        return 0
    elif x in U2R_type:
        return 0
    else:
        return 1


def data_process(data, rate=0):
    data_1_type = data[1].unique().tolist()
    data_2_type = data[2].unique().tolist()
    data_3_type = data[3].unique().tolist()
    data = data.drop([42], axis=1)  # 删除Level
    data[1] = data[1].apply(lambda x: data_1_type.index(x))
    data[2] = data[2].apply(lambda x: data_2_type.index(x))
    data[3] = data[3].apply(lambda x: data_3_type.index(x))
    data[41] = data[41].apply(lambda x: type_sort(x))
    # train_data.info()
    # train_data[41].unique()

    # data = shuffle(data)
    # 独热编码
    # lb = preprocessing.LabelBinarizer()
    # res = lb.fit_transform(data[41])
    res = data[41]
    unlabeled_data = data.drop([41], axis=1)
    if rate > 0:
        unlabeled_data, _, res, __ = train_test_split(unlabeled_data, res, test_size=1 - rate)
    return (unlabeled_data, res)

def get_train_data():
    train_data = pd.read_csv('./nsl-kdd/KDDTrain+_20Percent.txt', header=None)
    train_data, train_res = data_process(train_data)
    # print(train_data,train_res)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    train_data = pd.DataFrame(scaler.transform(train_data))
    # print(train_data)
    return train_data, train_res

def get_val_data(ini):
    val_data = pd.read_csv('./nsl-kdd/KDDTest+.txt',header=None)
    val_data = val_data[ini]
    (val_data, val_res) = data_process(val_data)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(val_data)
    val_data = pd.DataFrame(scaler.transform(val_data))
    return val_data, val_res

