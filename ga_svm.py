from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from functools import cmp_to_key
import copy
from ga import *
from data import *
from sklearn import svm
import svm_rl

pop_size = 500
max_gen = 20

class svm_individual:
    def __init__(self, bin_len):
        self.bin_code = gen(bin_len)
        self.fitness = 0
        self.status = []
        for x in range(0, len(self.bin_code)):
            if self.bin_code[x] == '1':
                self.status.append(x)

    def update_status(self):
        self.status = []
        for x in range(0, len(self.bin_code)):
            if self.bin_code[x] == '1':
                self.status.append(x)

    def output(self):
        print('bin_code:', self.bin_code)
        print('status:', self.status)
        print('fitness:', self.fitness)


class mytrack:
    def __init__(self):
        self.pop = []
        self.acc = []
        self.smi = []
        self.fit = []
        self.times = 0
#返回选择的数据属性值
def getdata(data, res, indi):
    indi.update_status()
    lo_data = data[indi.status]
    val_data1 = val_data[indi.status]
# 这个地方是两个结合的地方
def svm_get_fitness(data, res, indi, debug=False):
    indi.update_status()
    lo_data = data[indi.status]
    val_data1 = val_data[indi.status]
    # knn = KNeighborsClassifier(n_jobs=-1)
    # cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=None)
    # p = cross_val_score(knn, lo_data, res, cv=cv)
    # if debug:
    #     lo_data.info()
    #     print(p)
    #     print('\n----------------------')
    # return p.mean()


    clf = svm.LinearSVC(C=1, loss='hinge')
    clf.fit(lo_data, res)
    score = clf.score(val_data1, val_res)
    #print(score)
    return score



# 感觉选择和交叉算法可以在改一下
def gasvm_algo(pop, max_gen=30):
    track = mytrack()
    max_f = max_fit(pop)
    pop_size = len(pop)
    tmp_best = []
    best_1 = copy.deepcopy(sort_indi(pop)[0])
    best_2 = copy.deepcopy(sort_indi(pop)[1])
    while max_gen > 0:
        best_indi_1 = copy.deepcopy(best_1)
        best_indi_2 = copy.deepcopy(best_2)
        print('best_indi_1:\n')
        best_indi_1.output()
        print('best_indi_2:\n')
        best_indi_2.output()
        print('-----------------------------------')
        round_gen = []
        round_gen.append(best_indi_1)
        round_gen.append(best_indi_2)
        while len(round_gen) < pop_size:
            [parent_1, parent_2] = random_select(pop)
            [child_1, child_2] = crossover(parent_1, parent_2, rate=0.7)
            child_1 = mutate(child_1)
            child_2 = mutate(child_2)
            #这里结合的也要改
            getdata(child_1)
            getdata(child_2)
            child_1.fitness = svm_rl.getfitness(child_1)
            child_2.fitness = svm_rl.getfitness(child_2)
            #child_1.fitness = svm_get_fitness(train_data, train_res, child_1)
            #child_2.fitness = svm_get_fitness(train_data, train_res, child_2)
            round_gen.append(child_1)
            round_gen.append(child_2)
            round_gen = list(filter(None.__ne__, round_gen))
        mix_co = [sort_indi(round_gen)[0], sort_indi(round_gen)[1], best_indi_1, best_indi_2]
        best_1 = copy.deepcopy(sort_indi(mix_co)[0])
        best_2 = copy.deepcopy(sort_indi(mix_co)[1])
        pop = round_gen
        tmp_best.append(max_fit(pop))
        track.acc.append(best_1.fitness)
        max_gen = max_gen - 1
        print(30 - max_gen, 'finished!')
        print('------------------------------------------')
        print('\n')
    track.pop = tmp_best
    return track


times = 0
svm_choro = []

while times < pop_size:
    tmp = svm_individual(41)
    tmp.fitness = svm_get_fitness(train_data, train_res, tmp)
    svm_choro.append(tmp)
    times = times + 1
ta = gasvm_algo(svm_choro, max_gen)
fea = [0] * 41

for x in ta.pop:
    for y in x.status:
        fea[y] += 1

final_stat = []
for x in range(0, 41):
    if fea[x] > max_gen / 2:
        final_stat.append(x)
print(final_stat)
