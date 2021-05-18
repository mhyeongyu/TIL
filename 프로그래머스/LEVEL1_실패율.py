def solution(N, stages):
    dic = {}
    idx = 0
    length = len(stages)
    stages.sort()
    
    for n in range(1,N+1):
        cnt = 0
        people = length-idx
        
        while idx <= length-1:
            if stages[idx] > n: break
            else: cnt += 1
            idx += 1
            
        if cnt == 0: x = 0
        else: x = round((cnt/people), 20)
        dic[n] = x
        
    dic = sorted(dic, key=lambda x: dic[x], reverse=True)
    
    return dic