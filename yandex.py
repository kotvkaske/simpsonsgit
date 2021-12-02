def functionbar(N, S):
    l = []
    # for i in range(N):
    #     l.append(input())
    l = [3,2,1]
    l = sorted([int(x) for x in l])
    if N <= S:
        return 'INF'
    else:
        x_final = 0
        Flag = True
        x = 0
        while (Flag == True):
            x += 1
            x_check = S
            queue = []
            count = 0
            for i in (l):
                if count < S:
                    queue.append(i)
                    count += 1
                    continue

                if (count == S):
                    res = queue.pop(0)
                    if (res + x <= i):
                        queue.append(i)
                        x_check += 1
                    else:
                        break
            if x_check == S:
                x_final += 1
            elif x_check < S:
                Flag = False
                break

    if x_final != 0:
        return x_final
    else:
        return 'Impossible'


N, S = map(int, input().split())
print(functionbar(N, S))