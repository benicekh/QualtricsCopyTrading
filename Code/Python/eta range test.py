import numpy as np
from math import log

# Define the step size and the maximum value for the array
step = 0.1
max_value = 20.0

# Define the specific eta value
eta = 2.0

# Calculate the number of elements
num_elements = int(max_value / step) + 1

# Create an array of zeros with the required shape and dtype=object
array_2d = np.zeros((num_elements, num_elements), dtype=object)

# Create the range of values
range_values = np.arange(0, max_value + step, step)

# Assign the range to the first row and the first column
array_2d[0, :] = range_values
array_2d[:, 0] = range_values

def u(x, element):
    if x != 0:
        if element < 1 and element != 1:
            num = ((x**(1 - element))-1)
            den = (1 - element)
        elif element == 1:
            num = log(x)
            den = 1 
        else:
            num = ((x**(1 - element))-1)
            den = (1 - element)              
        return num / den
    else:
        if element != 1:
            return 0
        else:
            return float('-inf')  # Return negative infinity for log(0) case 
        
def calcEU(x,y,eta):
    return (0.5 * u(x, eta)) + (0.5 * u(y, eta))

# Iterate over the array and calculate the EU for the given eta
max_eu_value = float('-inf')
max_x = None
max_y = None

for i in range(1, num_elements):
    for j in range(1, num_elements):
        current_value = calcEU(array_2d[i, 0], array_2d[0, j], eta)
        if current_value > max_eu_value:
            max_eu_value = current_value
            max_x = array_2d[i, 0]
            max_y = array_2d[0, j]

# Print the maximum EU value and the corresponding x, y, eta
print(f"Maximum EU Value: {max_eu_value}")
print(f"Corresponding x: {max_x}, y: {max_y}, eta: {eta}")

-1.5-20
0-17.5
1-15
3-12.5
6-10

for y in np.arange(0.1, 20.1, 0.1):
    value = calcEU(10,(y-0.1),4.5)-calcEU(10,y,4.5)
    print(round(y,2))
    print (value)