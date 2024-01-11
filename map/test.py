import math

from sympy import *
x,y = symbols('x y')
expr = x+y
a = 10
b = 15
values = {x:a,y:b}
print(expr.subs(values))



print(math.sqrt((20.67-30)**2+(18-5)**2))