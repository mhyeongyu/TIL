#예산
def solution(d, budget):
    s = sum(d)
    if s <= budget:
        return len(d)

    d.sort()
    num = 0
    idx = 0
    while True:
        num += d[idx]
        idx += 1
        if num >= budget: break
        
    if num == budget: return idx
    else: return idx-1

#음양 더하기
def solution(absolutes, signs):
    length = len(absolutes)
    lst = []
    
    for idx in range(length):
        sign = signs[idx]
        if sign == True: n = absolutes[idx]
        else: n = -absolutes[idx]
        lst.append(n)
    
    return sum(lst)
