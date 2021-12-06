# 遗传算法 实数编码
import random
#初始化总群
class svm_ga:
    #n 个体数，pop总群 maxV个体最大边界值 step实数编码跨度
    def init_pop(n,pop,maxV, minV,step):
        #种群
        pop = []
        #个体
        for _ in range(n):
            r = random.random() #返回 [0.0, 1.0) 范围内的下一个随机浮点数
            individual = r*(maxV-minV)+maxV
            pop.append(individual)

