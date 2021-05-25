def solution(n):
    cnt = 1
    
    for i in range(1, n//2+1):
        sums = 0
        num = i
        
        while sums <= n:
            sums += num
            if sums == n:
                cnt += 1
                break
            num += 1
        
    return cnt