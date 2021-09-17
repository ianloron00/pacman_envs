import numpy as np

class Trial:
	def __init__(self, num):
		self.num = num

	def sum(self, a):
		return a + self.num

def fun_test(trial, val):
	trial.num = trial.sum(val)
	print(trial.num)
	return trial.num

trial = Trial(10)
fun_test(trial, 6)
print(type(trial))
