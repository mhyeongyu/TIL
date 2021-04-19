#소수만들기
def solution(nums):
    s = sum(nums)
    num = []
    for i in range(7, s+1):
        print(i)
        cnt = 0
        for j in range(1, i//2):
            if i%j == 0: cnt += 1
        if cnt == 1:
            num.append(i)

    ans = 0
    length = len(nums)
    for i in range(length):
        for j in range(i+1, length):
            for k in range(j+1, length):
                if nums[i] + nums[j] + nums[k] in num:
                    ans += 1
    return ans

#폰켓몬
def solution(nums):
    nums.sort()
    idx = 0
    num = []
    num.append(nums[0])
    for i in range(len(nums)-1):
        if nums[idx] != nums[idx+1]:
            num.append(nums[idx+1])
        idx += 1
    if len(num) > len(nums)/2: return len(nums)/2
    else: return len(num)