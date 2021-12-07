# 遗传算法 二进制编码
import random
import math
#初始化总群

#n 个体数，pop总群 maxV个体最大边界值 step实数编码跨度
def init_pop(n,maxV, minV):
    #种群
    pop = []
    #编码
    # c的取值大概在1-100吧
    N = math.ceil(math.sqrt(maxV-minV))
    #初始化个体
    for _ in range(n):
        r = random.random() #返回 [0.0, 1.0) 范围内的下一个随机浮点数
        individual = int(r*(maxV-minV)+maxV)
        bin_indi = bin(individual)  # 整形转换成二进制是以字符串的形式存在的 '0b10111110'
        #for j in range(len(bin_indi) - 2, N):  # 序列长度不足补0
        bin_indi ='0'*(N-len(bin_indi)) + bin_indi[2:] #序列长度不足补0
        pop.append(bin_indi)
    return pop

#交叉
def crossover(par_1, par_2, rate=0.8):
    pop_cro = par_1


if __name__ == '__main__':
    pop = init_pop(10,100,1)
    print(pop)




