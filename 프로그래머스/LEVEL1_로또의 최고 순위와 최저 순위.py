def solution(lottos, win_nums):
    lottos.sort()
    win_nums.sort()
    max_cnt = 0
    min_cnt = 0
    for i in lottos:
        if i == 0: max_cnt += 1
        else:
            for j in win_nums:
                if i == j: 
                    max_cnt += 1
                    min_cnt += 1
                else: continue
    
    high_rank = 7 - max_cnt
    low_rank = 7 - min_cnt
    
    if high_rank >= 6:
        high_rank = 6
    if low_rank >= 6:
        low_rank = 6
        
    return [high_rank, low_rank]