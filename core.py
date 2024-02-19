import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Scanner
def check_card_corners(card_corners):
    array = card_corners
    center = (round((array[0][0][0]+array[2][0][0])/2), round((array[0][0][1]+array[2][0][1])/2))
    for i in range(array.shape[0]):
        # print(point)
        if array[i][0][0] <= center[0] and array[i][0][1] <= center[1]:
            x0 = [[array[i][0][0], array[i][0][1]]]
        if array[i][0][0] <= center[0] and array[i][0][1] >= center[1]:
            x1 = [[array[i][0][0], array[i][0][1]]]
        if array[i][0][0] >= center[0] and array[i][0][1] <= center[1]:
            x3 = [[array[i][0][0], array[i][0][1]]]
        if array[i][0][0] >= center[0] and array[i][0][1] >= center[1]:
            x2 = [[array[i][0][0], array[i][0][1]]]
    array = np.array([x0,x1,x2,x3])
    return array

# Crop the feature
def cropFeature(image):
    # image = cv2.resize(image, (300, 200))
    set_position = {'number': [140*3, 41*3, 280*3, 69*3],
                    'name': [91*3, 59*3, 298*3, 105*3],
                    'dob': [90*3, 100*3, 295*3, 125*3],
                    'hometown': [89*3, 121*3, 295*3, 163*3],
                    'address': [88*3, 159*3, 297*3, 198*3]}

    for key, position in set_position.items():
        x1, y1, x2,y2 = position

        # Specify the color of the rectangle in BGR format (Blue, Green, Red)
        color = (0, 255, 0)  # This is a green rectangle

        # Draw the rectangle on the image
        # cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        if key == "number":
            number_image = image[y1:y2,x1:x2,]
        elif key == "name":
            name_image = image[y1:y2,x1:x2,]
        elif key == "dob":
            dob_image = image[y1:y2,x1:x2,]
        elif key == "hometown":
            hometown_image = image[y1:y2,x1:x2,]
        elif key == "address":
            address_image = image[y1:y2,x1:x2,]
        # print(key, position )
    return {'number':number_image,
            'name':name_image,
            'dob':dob_image,
            'hometown': hometown_image,
            'address':address_image}

def warpImage(image, dst):
    card_warp = np.zeros((600, 900, 3), dtype=np.uint8)
    target_corners = np.array([[0, 0], [0,599], [899, 599], [899,0]], dtype=np.float32)
    # global transformation_matrix
    transformation_matrix = cv2.getPerspectiveTransform(dst.astype(np.float32), target_corners)
    card_warp = cv2.warpPerspective(image, transformation_matrix, (900, 600))
    # plt.imshow(card_warp)
    return card_warp

def angle_between_points(point1, point2, point3):
    vector1 = point1 - point2
    vector2 = point3 - point2

    angle_rad = np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0])
    angle_deg = np.degrees(angle_rad)

    # Ensure the angle is between 0 and 360 degrees
    angle_deg = (angle_deg + 360) % 360

    return angle_deg

def convert_points_array(points_array):
    # Reshape the array to (num_points, 2)
    reshaped_array = np.reshape(points_array, (points_array.shape[0], 2))
    return reshaped_array

def check_angle_in_range(list_points, coordinate_range=(81, 99)):
    for point in list_points:
      if point <81 or point >99:
        return False
    return True

def check_number_point_in_zone(list_number_points):
    count = 0 
    for point in list_number_points:
      if point == 0:
        count += 1
    return count <= 1