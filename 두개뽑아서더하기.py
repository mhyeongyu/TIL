def solution(numbers):
    ans = []
    length = len(numbers)
    for i in range(length):
        for j in range(i+1, length):
            num = numbers[i] + numbers[j]
            if num not in ans:
                ans.append(num)
    ans.sort()
    return ans