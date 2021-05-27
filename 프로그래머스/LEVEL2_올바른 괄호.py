def solution(s):
    if s[0] == ')': return False
    elif s[-1] == '(': return False
    else:
        cnt = 0
        for i in s:
            if cnt < 0: break
            if i == '(': cnt += 1
            else: cnt -= 1
        
        if cnt == 0: return True
        else: return False