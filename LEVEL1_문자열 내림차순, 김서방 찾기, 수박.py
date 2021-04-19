#문자열 내림차순 배치
def solution(s):
    lst = []
    for i in s:
        lst.append(i)
    lst.sort(reverse=True)
    ans = ''.join(lst)
    return ans

#서울에서 김서방 찾기
def solution(seoul):
    idx = 0
    while True:
        if seoul[idx] == 'Kim': break
        idx += 1
    return "김서방은 {0}에 있다".format(idx)

#수박수박수박
def solution(n):
    cnt = 0
    lst = []
    while cnt < n:
        if cnt%2 == 0: lst.append('수')
        else: lst.append('박')
        cnt += 1
    return ''.join(lst)