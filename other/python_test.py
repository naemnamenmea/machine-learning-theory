print(1/32.)

# print(127 & (62^62) | 31)

# a = 91
# b = 14
# while(a!=b):
#     if a > b:
#         a = a-b
#     else:
#         b = b-a
#
# print(a)

# a = 10
# b = 5
# a = a ^ b
# b = a ^ b
# a = a ^ b
# print(a+b)

# Считать через запятую 2 переменные

# class Account:
#     def __init__(self,id):
#         self.id = id
#         id = 42
#
# a = Account(1)
# print(a.id)

#'a' + ' ' + 'a' + ..
# str = ''
# str.join()
# print(str)

#Типы, которые могут использоваться в словаре как ключи
# tuple list object dict frozenset set

# print(r"\nwoow")

# x = str()
# x = int()
# x = double() <--
# x = NoneType()

# a = [1,2]
# print(type(a))

# def dec(func):
#     print("X",end='')
#     def inner():
#         return func()
#     print("Y", end='')
#     return inner
# 
# @dec
# def z():
#     print("Z", end='')
#     return 0
# 
# print("T",end='')
# z()

# class User:
#     desc = "user"
#     id = 0
#     def __init__(self, id):
#         self.id = id
#
# u = User(1)
# User.desc = "user object"
# print(u.id, u.desc)

# x = input('x')
# print(x)

# d = dict()
# d['a'] += 1
# d['b'] = 1
# print(d.keys())

# def check(func):
#     def inner():
#         try:
#             return func()
#         except Exception:
#             return None
#     return inner
# 
# @check
# def div():
#     return 10/0
# 
# res = div()
# print(res)

# x = [1,2,3,4,5]
# f = lambda x: x+ [6]
# f(x)
# s = sum(x)
# print(s)