def solution(n):
    num = str(n)
    lst = []
    for i in num:
        lst.append(int(i))
    lst.sort(reverse=True)
    ans = ''
    for i in lst:
        ans += str(i)
    return int(ans)