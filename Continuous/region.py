

#This represents a rectangular region
class Rectangle():
	#requires left bottom point and right top point
	def __init__(self, x1, y1, x2, y2, outer_boundary=False):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.outer_boundary = outer_boundary


	#Returns True if the point is in the rectangle other wise False
	def is_in(self,x, y) -> bool:
		if self.outer_boundary:
			if x == self.x1 or x == self.x2 or y == self.y1 or y == self.y2:
				return False

		if x < self.x1:
			return False

		if x > self.x2:
			return False

		if y < self.y1:
			return False

		if y > self.y2:
			return False

		return True


	def get_x1(self):
		return self.x1

	def get_x2(self):
		return self.x2

	def get_y1(self):
		return self.y1

	def get_y2(self):
		return self.y2

	def length(self):
		return (self.x2 - self.x1)

	def width(self):
		return (self.y2 - self.y1)

	def center(self):
		return (self.x1 + self.x2)/2 , (self.y1 + self.y2)/2