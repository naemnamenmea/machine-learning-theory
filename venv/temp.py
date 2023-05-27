from ast import literal_eval
from itertools import product
from functools import reduce
import sys

def validate(numbers, result):
     if sum(numbers) >= result >= reduce(int.__sub__, sorted(numbers)):
         res = calculate(numbers, result)
         if res != 'No solution':
            return res+'='+str(result)
         else:
             return res
     return 'No solution'



def calculate(numbers, result):
     pattern = '{}'.join(map(str, numbers))
     for i in product(['+', '-'], repeat=len(numbers)-1):
         expr = pattern.format(*i)
         if eval(expr) == result:
             return expr
     return 'No solution'

lines = sys.stdin.readlines()
n, s = map(int,lines[0].split())
arr = [int(i) for i in lines[1].split()]
print(validate(arr, s))

# nums = [int(i) for i in input().split()]
# print(validate(nums[2:], nums[1]))

# print(validate([4, 2, 2, 1], 9))
