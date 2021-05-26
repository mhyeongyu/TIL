def solution(arr1, arr2):
    ans = []
    new_arr2 = []
    
    for i in range(len(arr2[0])):
        arr = []
        for a in arr2:
            arr.append(a[i])
        new_arr2.append(arr)
    
    le = len(arr1[0])
    for i in arr1:
        add_arr = []
        for j in new_arr2:
            num = 0
            for k in range(le):
                num += (i[k] * j[k])
            add_arr.append(num)
        ans.append(add_arr)
    
    return ans