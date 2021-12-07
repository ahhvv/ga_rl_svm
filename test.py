import T_ga1
import random
# math.sin(math.radians(90))
# math.sin 是以弧度作为单位的
# 求 cos(x)+sin(x) 最大值
if __name__ == '__main__':
    step = 10
    n = 10
    maxV = -1
    minV = 1
    pop = T_ga1.init_pop(n, maxV, minV)
    L = len(pop)
    #随机选择2个交叉
    for _ in range(step):
        for _ in range(L):
            a_index = random.randint(0, L-1)
            b_index = random.randint(0, L-1)
            a, b = T_ga1.crossover(pop[a_index], pop[b_index], rate_cro=0.8)
            pop[a_index] = a
            pop[b_index] = b
        # 变异
        for i in range(L):
            pop[i] = T_ga1.mutation(pop[i])
        pop = T_ga1.select(pop)
    print(pop)

