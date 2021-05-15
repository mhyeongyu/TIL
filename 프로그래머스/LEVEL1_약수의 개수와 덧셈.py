import math

def solution(left, right):
    ans = 0
    
    for i in range(left, right+1):
        root_i = int(math.sqrt(i))
        if i == (root_i ** 2): ans -= i
        else: ans += i
    
    return ans