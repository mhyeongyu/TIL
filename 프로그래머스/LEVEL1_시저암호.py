def solution(s, n):
    up = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lower = 'abcdefghijklmnopqrstuvwxyz'
    ans = ''
    for i in s:
        if i in up:
            idx = up.find(i) + n
            ans += up[idx%26]
        elif i in lower:
            idx = lower.find(i) + n
            ans += lower[idx%26]
        else:
            ans += ' '
    return ans