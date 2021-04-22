def solution(numbers, hand):
    
    left_number = [1, 4, 7]
    right_number = [3, 6, 9]
    
    left_loc = [3, 0]
    right_loc = [3, 2]
    lst = []
    
    for i in numbers:
        if i in left_number:
            lst.append('L')
            left_loc = [i//3, 0]
        elif i in right_number:
            lst.append('R')
            right_loc = [i//3 -1, 2]
        else:
            if i == 0: middle = [3, 1]
            else: middle = [i//3, 1]
            
            ml = abs(middle[0] - left_loc[0]) + abs(middle[1] - left_loc[1])
            mr = abs(middle[0] - right_loc[0]) + abs(middle[1] - right_loc[1])
            
            if ml > mr: 
                lst.append('R')
                right_loc = middle
            elif ml < mr: 
                lst.append('L')
                left_loc = middle
            else:
                if hand == 'right': 
                    lst.append('R')
                    right_loc = middle
                else: 
                    lst.append('L')
                    left_loc = middle
    
    ans = ''.join(lst)
    return ans