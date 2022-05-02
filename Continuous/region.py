

#This represents a rectangular region
class Rectangle():
	#requires left bottom point and right top point
	def __init__(self, x1, y1, x2, y2):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2


	#Returns True if the point is in the rectangle other wise False
	def is_in(self,x, y) -> bool:
		if x < self.x1:
			return False

		if x > self.x2:
			return False

		if y < self.y1:
			return False

		if y > self.y2:
			return False

		return True

