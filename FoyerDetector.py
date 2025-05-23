from shapely.geometry import Point, Polygon

class FoyerDetector:
    def __init__(self):
        # Define foyer boundaries (example coordinates)
        #x,z
        self.foyer_polygon = Polygon([

            (-1.3, -0.8),   # Flipped from (-0.75, -1.3)
            (-1.20, 0.85),    # Flipped from (0.8, -1.20)
            (-0.5, 0.85),    # Flipped from (0.75, -0.5)
            (-0.65, -0.9)

            
            #(-0.75, -1.3),    # Bottom-left corner
            #(0.8, -1.20),    # Bottom-right corner
            #(0.75, -0.5),    # Top-right corner
            #(-0.85, -0.65)     # Top-left corner
        ])
        self.was_in_foyer = True
    #FIXME can refactor to pass in a point instead of a list for current_position
    def is_in_foyer(self, point):
        """Check if a point is inside the foyer"""
        #print("Foyer boolean")
        #print(self.foyer_polygon.contains(point))
        return self.foyer_polygon.contains(point)

    def has_left_foyer(self, current_position):
        """Check if subject has left the foyer area"""
        is_in_foyer = self.is_in_foyer(current_position)
        #print(current_position)
        #print("Current Position^^^")
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
