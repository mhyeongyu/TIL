def solution(n, arr1, arr2):
    ans = []
    for i in range(n):
        num1 = arr1[i] #9
        num2 = arr2[i] #30
        a = n-1
        b = n-1
        
        num1_lst = []
        num2_lst = []
        
        while a != -1:
            if num1 >= 2 ** a:
                num1_lst.append(1)
                num1 -= 2 ** a
                a -= 1
            else:
                num1_lst.append(0)
                a -= 1
        
        while b != -1:
            if num2 >= 2 ** b:
                num2_lst.append(1)
                num2 -= 2 ** b
                b -= 1
            else:
                num2_lst.append(0)
                b -= 1 
        st = ''
        for i in range(n):
            if num1_lst[i] == 1 or num2_lst[i] == 1:
                st += '#'
            else:
                st += ' '
            
        ans.append(st)
    return ans