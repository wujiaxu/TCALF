from typing import List
from shapely.geometry import MultiPolygon, LinearRing,Point

class Map:
    obstacles:List
    def __init__(self,map_size):
        self._map_size = map_size
        #consider the map shape is square in default
        self._map_boundary = LinearRing( ((-self._map_size/2., self._map_size/2.), 
                                    (self._map_size/2., self._map_size/2.),
                                    (self._map_size/2., -self._map_size/2.),
                                    (-self._map_size/2.,-self._map_size/2.)) )
        
    def getBoundary(self):

        return self._map_boundary.coords.xy