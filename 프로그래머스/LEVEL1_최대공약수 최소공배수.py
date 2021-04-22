def solution(n, m):
    if n > m: cnt = m
    else: cnt = n
    
    num = 2
    ab = 1

    while num <= cnt:
        if n%num == 0 and m%num == 0:
            n //= num
            m //= num
            ab *= num
        else: num += 1
    
    return [ab, ab*n*m]