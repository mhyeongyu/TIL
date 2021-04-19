#내적
def solution(a, b):
    idx = 0
    num = 0
    for i in range(len(a)):
        n = a[idx] * b[idx]
        idx += 1
        num += n
    return num

#문자열 내 p와 y의 개수
def solution(s):
    low_s = s.lower()
    p_cnt = 0
    y_cnt = 0
    for i in low_s:
        if i == 'p': p_cnt += 1
        elif i == 'y': y_cnt += 1
    if p_cnt == y_cnt: return True
    else: return False
