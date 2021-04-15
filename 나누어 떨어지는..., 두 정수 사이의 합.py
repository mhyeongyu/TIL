#나누어 떨어지는 숫자 배열
def solution(arr, divisor):
    ans = []
    for i in arr:
        div = i % divisor
        if div == 0: ans.append(i)
    
    if ans == []: ans.append(-1)
    ans.sort()
    return ans

#두 정수 사이의 합
def solution(a, b):
    num = 0
    if a > b: 
        for i in range(b, a+1): 
            num += i
        return num
    elif a < b:
        for i in range(a, b+1): 
            num += i
        return num
    else: return a