import random
from functools import cmp_to_key

def gen(x):
    return ''.join([random.choice(['0', '1']) for _ in range(x)])


def cmp(a, b):
    if a > b:
        return 1
    elif a < b:
        return -1
    else:
        return 0


def max_fit(pop):
    tmp = sorted(pop, key=cmp_to_key(lambda x, y: cmp(x.fitness, y.fitness)), reverse=True)
    return tmp[0]


def output_indi(pop, num):
    tmp = sorted(pop, key=cmp_to_key(lambda x, y: cmp(x.fitness, y.fitness)), reverse=True)
    for x in range(0, num):
        print('no.' + str(x))
        tmp[x].output()
        print('------------------------------------')


def sort_indi(pop):
    tmp = sorted(pop, key=cmp_to_key(lambda x, y: cmp(x.fitness, y.fitness)), reverse=True)
    return tmp

# 这个选择方法可以换一下，轮盘赌什么的
def random_select(pop):
    all_indi = 0
    for x in pop:
        all_indi += x.fitness
    rtn = []
    while len(rtn) < 2:
        for x in range(0, len(pop)):
            if random.random() < pop[x].fitness / all_indi:
                rtn.append(pop[x])
                if len(rtn) > 1:
                    break
    return rtn

# 点交叉
def crossover(par_1, par_2, rate=0.8, min_num=3, max_num=10):
    cross_pos = []
    child_1 = par_1
    child_2 = par_2

    child_1.bin_code = list(child_1.bin_code)
    child_2.bin_code = list(child_2.bin_code)

    for x in range(0, len(par_1.bin_code)):
        if random.random() < rate:
            cross_pos.append(x)

    for x in cross_pos:
        child_1.bin_code[x] = par_2.bin_code[x]
        child_2.bin_code[x] = par_1.bin_code[x]

    child_1.bin_code = ''.join(child_1.bin_code)
    child_2.bin_code = ''.join(child_2.bin_code)
    return [child_1, child_2]


def mutate(indi, rate=0.02):
    indi.bin_code = list(indi.bin_code)
    for x in range(0, len(indi.bin_code)):
        if random.random() < rate:
            if indi.bin_code[x] == '0':
                indi.bin_code[x] == '1'
            else:
                indi.bin_code[x] == '0'
    indi.bin_code = ''.join(indi.bin_code)
    return indi
