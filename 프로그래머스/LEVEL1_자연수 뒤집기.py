def solution(n):
    lst = []
    str_n = str(n)
    idx = 0
    for i in range(len(str_n)):
        num = str_n[idx-1]
        lst.append(int(num))
        idx -= 1
    return lst