import math
import numpy as np

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

if __name__=="__main__":

    vertices = genTriangle((0,0),(2*np.pi/3.,2*np.pi/3.),2)
    print(vertices)