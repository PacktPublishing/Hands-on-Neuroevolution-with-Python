#
# Here we define common geometric primitives along with utilities
# allowing to find distance from point to the line, to find intersection point
# of two lines, and to find the length of the line in two dimensional Euclidean 
# space.
#

import math

def deg_to_rad(degrees):
    """
    The function to convert degrees to radians.
    Arguments:
        degrees: The angle in degrees to be converted.
    Returns:
        The degrees converted to radians.
    """
    return degrees / 180.0 * math.pi

def read_point(str):
    """
    The function to read Point from specified string. The point
    coordinates are in order (x, y) and delimited by space.
    Arguments:
        str: The string encoding Point coorinates.
    Returns:
        The Point with coordinates parsed from provided string.
    """
    coords = str.split(' ')
    assert len(coords) == 2
    return Point(float(coords[0]), float(coords[1]))

def read_line(str):
    """
    The function to read line segment from provided string. The coordinates
    of line end points are in order: x1, y1, x2, y2 and delimited by spaces.
    Arguments:
        str: The string to read line coordinates from.
    Returns:
        The parsed line segment.
    """
    coords = str.split(' ')
    assert len(coords) == 4
    a = Point(float(coords[0]), float(coords[1]))
    b = Point(float(coords[2]), float(coords[3]))
    return Line(a, b)

class Point:
    """
    The basic class describing point in the two dimensional Cartesian coordinate
    system.
    """
    def __init__(self, x, y):
        """
        Creates new point at specified coordinates
        """
        self.x = x
        self.y = y

    def angle(self):
        """
        The function to determine angle in degrees of vector drawn from the
        center of coordinates to this point. The angle values is in range
        from 0 to 360 degrees in anticlockwise direction.
        """
        ang = math.atan2(self.y, self.x) / math.pi * 180.0
        if (ang < 0.0):
            # the lower quadrants (3 or 4)
            return ang + 360
        return ang

    def rotate(self, angle, point):
        """
        The function to rotate this point around another point with given
        angle in degrees.
        Arguments:
            angle: The rotation angle (degrees)
            point: The point - center of rotation
        """
        rad = deg_to_rad(angle)
        # translate to have another point at the center of coordinates
        self.x -= point.x
        self.y -= point.y
        # rotate
        ox, oy = self.x, self.y
        self.x = math.cos(rad) * ox - math.sin(rad) * oy
        self.y = math.sin(rad) * ox - math.cos(rad) * oy
        # restore
        self.x += point.x
        self.y += point.y

    def distance(self, point):
        """
        The function to caclulate Euclidean distance between this and given point.
        Arguments:
            point: The another point
        Returns:
            The Euclidean distance between this and given point.
        """
        dx = self.x - point.x
        dy = self.y - point.y

        return math.sqrt(dx*dx + dy*dy)
    
    def __str__(self):
        """
        Returns the nicely formatted string representation of this point.
        """
        return "Point (%.1f, %.1f)" % (self.x, self.y)

class Line:
    """
    The simple line segment between two points. Used to represent maze wals.
    """
    def __init__(self, a, b):
        """
        Creates new line segment between two points.
        Arguments:
            a, b: The end points of the line
        """
        self.a = a
        self.b = b

    def midpoint(self):
        """
        The function to find midpoint of this line segment.
        Returns:
            The midpoint of this line segment.
        """
        x = (self.a.x + self.b.x) / 2.0
        y = (self.a.y + self.b.y) / 2.0

        return Point(x, y)

    def intersection(self, line):
        """
        The function to find intersection between this line and the given one.
        Arguments:
            line: The line to test intersection against.
        Returns:
            The tuple with the first value indicating if intersection was found (True/False)
            and the second value holding the intersection Point or None
        """
        A, B, C, D = self.a, self.b, line.a, line.b

        rTop = (A.y - C.y) * (D.x - C.x) - (A.x - C.x) * (D.y - C.y)
        rBot = (B.x - A.x) * (D.y - C.y) - (B.y - A.y) * (D.x - C.x)

        sTop = (A.y - C.y) * (B.x - A.x) - (A.x - C.x) * (B.y - A.y)
        sBot = (B.x - A.x) * (D.y - C.y) - (B.y - A.y) * (D.x - C.x)

        if rBot == 0 or sBot == 0:
            # lines are parallel
            return False, None

        r = rTop / rBot
        s = sTop / sBot
        if r > 0 and r < 1 and s > 0 and s < 1:
            x = A.x + r * (B.x - A.x)
            y = A.y + r * (B.y - A.y)
            return True, Point(x, y)

        return False, None

    def distance(self, p):
        """
        The function to estimate distance to the given point from this line.
        Arguments:
            p: The point to find distance to.
        Returns:
            The distance between given point and this line.
        """
        utop = (p.x - self.a.x) * (self.b.x - self.a.x) + (p.y - self.a.y) * (self.b.y - self.a.y)
        ubot = self.a.distance(self.b)
        ubot *= ubot
        if ubot == 0.0:
            return 0.0

        u = utop / ubot
        if u < 0 or u > 1:
            d1 = self.a.distance(p)
            d2 = self.b.distance(p)
            if d1 < d2:
                return d1
            return d2
        
        x = self.a.x + u * (self.b.x - self.a.x)
        y = self.a.y + u * (self.b.y - self.a.y)
        point = Point(x, y)
        return point.distance(p)

    def length(self):
        """
        The function to calculate the length of this line segment.
        Returns:
            The length of this line segment as distance between its endpoints.
        """
        return self.a.distance(self.b) 

    def __str__(self):
        """
        Returns the nicely formatted string representation of this line.
        """
        return "Line (%.1f, %.1f) -> (%.1f, %.1f)" % (self.a.x, self.a.y, self.b.x, self.b.y)