def solution(board, moves):
    cnt = 0
    length = len(board)
    ans = []
    
    for i in moves:
        idx = 0
        while idx < length:
            num = board[idx][i-1]
            if num == 0: 
                idx += 1
            else:
                board[idx][i-1] = 0
                if ans == []: 
                    ans.append(num)
                    break
                else:
                    if ans[-1] == num: 
                        ans.pop()
                        cnt += 2
                    else:
                        ans.append(num)
                    break
    return cnt