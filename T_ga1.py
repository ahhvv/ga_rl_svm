# 遗传算法 实数编码
import random
import math
import numpy as np
#初始化总群
#n 个体数，pop总群 maxV，minV个体最大小边界值
def init_pop(n,maxV, minV):
    #种群
    pop = []
    for _ in range(n):
        r = random.random()  # 返回 [0.0, 1.0) 范围内的下一个随机浮点数
        individual = (r * (maxV - minV) + minV)
        pop.append(individual)
    return pop
#交叉
def crossover(par_1, par_2, rate_cro=0.8):
    child_1 = par_1
    child_2 = par_2
    #% 交叉算子
    r = random.random()
    if (r < rate_cro):
        r1 = random.random()
        child_1 = r1 * par_1 + (1 - r1) * par_1;
        child_2 = r1 * par_2 + (1 - r1) * par_1;
    return child_1, child_2
#变异
def mutation(par, rate_mut=0.1, step = 1 ):
    r = random.random()
    if (r < rate_mut):
        r1 = random.uniform(-1, 1)*step
        par = par + r1
    return par
# 获得适应度 接口
def getfitness(individual):
    #返回每个体适应度
    individual =math.sin(math.radians(individual))+math.cos(math.radians(individual))
    return individual
#选择 返回一个个体
def select(pop):
    # 全体适应度
    fitness = []
    """根据轮盘赌法选择优秀个体"""
    for i in range(len(pop)):
        fit = getfitness(pop[i])+2
        fitness.append(fit)
    fitness = fitness / np.sum(fitness)  # 归一化
    l = len(fitness)
    # 根据概率选择 #replace:True表示可以取相同数字，False表示不可以取相同数字
    pop1 = np.random.choice(pop, size=l, replace=True, p=fitness)
    return pop1



