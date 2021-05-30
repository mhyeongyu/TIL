def solution(prices):
    ans = []
    length = len(prices)
    
    for i in range(length-1):
        distance = i + 1
        
        while distance < length:
            if prices[i] > prices[distance]:
                ans.append(distance - i)
                break
            else:
                distance += 1
                
        if distance == length:
            ans.append(distance - i - 1)
    ans.append(0)
    
    return ans