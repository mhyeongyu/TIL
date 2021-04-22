def solution(n):
    cnt = 1
    while cnt < n//2:
        if cnt**2 == n: break
        cnt += 1
    if cnt**2 == n: return (cnt+1) **2
    else: return -1