def solution(n):
    num_lst = set(range(2, n+1))
    # print(num_lst)
    
    for i in range(2, n+1):
        if i in num_lst:
            num_lst -= set(range(2*i, n+1, i))
    # print(num_lst)
    return len(num_lst)