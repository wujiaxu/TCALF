import math
from networkx import center
import numpy as np
import typing as tp
from shapely.geometry import Polygon,MultiPolygon

def genTriangle(center, inner_angle, radius):

    # Define angles for each vertex
    angles = [np.pi/3., np.pi/3. + inner_angle[0], 
              np.pi/3. + inner_angle[0]+inner_angle[1]]
    
    # Initialize list to store vertices
    inscribed_points = []
    
    # Calculate vertices using trigonometry
    for angle in angles:
        x = center[0] + radius * math.cos(angle) 
        y = center[1] + radius * math.sin(angle) 
        inscribed_points.append((x, y))
    vertices = []

    for i in range(len(inscribed_points)):
        # Extract slope (k) and y-intercept (b) from the equations of the lines
        k1= -1./np.tan(angles[i])
        b1 = inscribed_points[i][1]- inscribed_points[i][0]*k1
        j = (i+1)%3
        k2= -1./np.tan(angles[j])
        b2 = inscribed_points[j][1]- inscribed_points[j][0]*k2
        
        # Calculate x-coordinate of the intersection point
        x_intersection = (b2 - b1) / (k1 - k2)
        
        # Calculate y-coordinate of the intersection point
        y_intersection = k1 * x_intersection + b1 

        vertices.append((x_intersection,y_intersection))
    
    return vertices

def genRectangle(x,y,w,h,yaw):
    rectangle = np.array([
        [-w/2.,h/2.],
        [w/2.,h/2.],
        [w/2.,-h/2.],
        [-w/2.,-h/2.]
    ])
    rectangle = np.array([x,y])\
                +rectangle.dot(np.array([[np.cos(yaw),np.sin(yaw)],
                           [-np.sin(yaw),np.cos(yaw)]]))
    
    return [(rectangle[i,0],rectangle[i,1]) for i in range(rectangle.shape[0])]

def genHexagon(x,y,radius,yaw):
    vertices = []
    for i in range(6):
        vertices.append([radius*np.cos(i*np.pi/3),radius*np.sin(i*np.pi/3.)])
    vertices = np.array([x,y])\
                +np.array(vertices).dot(np.array([[np.cos(yaw),np.sin(yaw)],
                           [-np.sin(yaw),np.cos(yaw)]]))
    return [(vertices[i,0],vertices[i,1]) for i in range(vertices.shape[0])]

def get_default_scenario(map_size:float)->tp.Union[Polygon,MultiPolygon]:
    #gen triangle
    x = (np.random.random() - 0.5)*map_size
    y = (np.random.random() - 0.5)*map_size
    angle1 = (np.random.random() - 0.5)*np.pi/3. + 2*np.pi/3.
    angle2 = (np.random.random() - 0.5)*np.pi/3. + 2*np.pi/3.
    radius = np.random.random() * map_size/8.
    triangle = genTriangle((x,y), [angle1,angle2], radius)
    #gen rectangle
    x = (np.random.random() - 0.5)*map_size
    y = (np.random.random() - 0.5)*map_size
    w = np.random.random() * map_size/6.
    h = np.random.random() * map_size/6.
    yaw = (np.random.random() - 0.5)*2*np.pi
    rectangle = genRectangle(x,y,w,h,yaw)
    #gen hex
    x = (np.random.random() - 0.5)*map_size
    y = (np.random.random() - 0.5)*map_size
    radius = np.random.random() * map_size/6.
    yaw = (np.random.random() - 0.5)*2*np.pi
    hex = genHexagon(x,y,radius,yaw)

    triangle = Polygon(triangle)
    rectangle = Polygon(rectangle)
    hex = Polygon(hex)

    boundary_polygon = triangle.union(rectangle).union(hex)

    return boundary_polygon

def get_hallway_scenario(map_size:float)->tp.Union[Polygon,MultiPolygon]:
    # width is set to be 1/3 ~ 1/4 map_size
    width = (np.random.random()*(1/3.-1/4.)+1/4.)*map_size
    wall_w = (map_size-width)/2.
    center_x_1 = (width+wall_w)/2.
    center_x_2 = -center_x_1
    wall_1 = genRectangle(center_x_1,0,wall_w,map_size,0)
    wall_2 = genRectangle(center_x_2,0,wall_w,map_size,0)
    wall_1 = Polygon(wall_1)
    wall_2 = Polygon(wall_2)
    boundary_polygon = wall_1.union(wall_2)
    return boundary_polygon

def get_doorway_scenario(map_size:float)->tp.Union[Polygon,MultiPolygon]:
    # width is set to be 1/2 ~ 1/3 map_size
    width = np.random.random()*(1.2-0.8)+0.8
    wall_w = (map_size-width)/2.
    center_x_1 = (width+wall_w)/2.
    center_x_2 = -center_x_1
    wall_1 = genRectangle(center_x_1,0,wall_w,0.2,0)
    wall_2 = genRectangle(center_x_2,0,wall_w,0.2,0)
    wall_1 = Polygon(wall_1)
    wall_2 = Polygon(wall_2)
    boundary_polygon = wall_1.union(wall_2)
    return boundary_polygon

def get_cross_scenario(map_size:float)->tp.Union[Polygon,MultiPolygon]:
    # width is set to be 1/2 ~ 1/3 map_size
    width = (np.random.random()*(1/3.-1/4.)+1/4.)*map_size
    wall_w = (map_size-width)/2.
    center = (width+wall_w)/2.
    wall_1 = genRectangle(center,center,wall_w,wall_w,0)
    wall_2 = genRectangle(center,-center,wall_w,wall_w,0)
    wall_1 = Polygon(wall_1)
    wall_2 = Polygon(wall_2)
    wall_3 = genRectangle(-center,-center,wall_w,wall_w,0)
    wall_4 = genRectangle(-center,center,wall_w,wall_w,0)
    wall_3 = Polygon(wall_3)
    wall_4 = Polygon(wall_4)
    boundary_polygon = wall_1.union(wall_2).union(wall_3).union(wall_4)
    return boundary_polygon

if __name__=="__main__":

    vertices = genTriangle((0,0),(2*np.pi/3.,2*np.pi/3.),2)
    print(vertices)