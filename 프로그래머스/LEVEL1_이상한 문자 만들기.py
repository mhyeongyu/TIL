def solution(s):
    lst = []
    split_s = s.split(' ')
    for i in split_s:
        idx = 0
        ans = ''
        for j in i:
            num = idx % 2
            if num == 0: a = j.upper()
            else: a = j.lower()
            ans += a
            idx += 1
        lst.append(ans)
    answer = ' '.join(lst)
    return answer