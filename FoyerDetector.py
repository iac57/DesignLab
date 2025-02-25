from shapely.geometry import Point, Polygon

class FoyerDetector:
    def __init__(self):
        # Define foyer boundaries (example coordinates)
        self.foyer_polygon = Polygon([
            (0, 0),    # Bottom-left corner
            (5, 0),    # Bottom-right corner
            (5, 5),    # Top-right corner
            (0, 5)     # Top-left corner
        ])
        self.was_in_foyer = True
    #FIXME can refactor to pass in a point instead of a list for current_position
    def is_in_foyer(self, point):
        """Check if a point is inside the foyer"""
        return self.foyer_polygon.contains(point)

    def has_left_foyer(self, current_position):
        """Check if subject has left the foyer area"""
        is_in_foyer = self.is_in_foyer(current_position)
        
        if self.was_in_foyer and not is_in_foyer:
            self.was_in_foyer = False
            return True
        return False

    def has_reentered_foyer(self, current_position):
        """Check if subject has reentered the foyer area"""
        is_in_foyer = self.is_in_foyer(current_position)
        
        if not self.was_in_foyer and is_in_foyer:
            self.was_in_foyer = True
            return True
        return False
