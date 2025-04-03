import sys
import random

sys.stdout = open("toyexample.in", "w")

print(3000, 3000)

for i in range(3000):
    for j in range(3000):
        print(random.random(), end=' ')
    print()