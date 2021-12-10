import T_ga1
import random
import matplotlib.pyplot as plt
import os
# math.sin(math.radians(90))
# math.sin 是以弧度作为单位的
# 求 cos(x)+sin(x) 最大值
if __name__ == '__main__':
    step = 100
    n = 10
    maxV = 0
    minV = 100
    pop = T_ga1.init_pop(n, maxV, minV)
    L = len(pop)
    all_bestIndi = []
    bestIndi = T_ga1.getBest(pop)

    for _ in range(step):
        # 随机选择2个交叉
        for j in range(L):
            a_index = random.randint(0, L-1)
            b_index = random.randint(0, L-1)
            a, b = T_ga1.crossover(pop[a_index], pop[b_index], rate_cro=0.8)
            pop[a_index] = a
            pop[b_index] = b
        # 变异
        for i in range(L):
            pop[i] = T_ga1.mutation(pop[i])
        pop = T_ga1.select(pop)
        # 把上次最好的个体保留起来
        pop[0] = bestIndi
    print(pop)

# 画出图片
    plt.plot(all_bestIndi, label='bestIndi')
    if not os.path.exists('image'):
        os.makedirs('image')
    plt.savefig(os.path.join('image', 'bestIndi' ))