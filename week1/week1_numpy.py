import math
import numpy as np

def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-1.0 * x))
    return s

# limitation of math packahge
print(basic_sigmoid(3))
x = np.array([1, 2, 3])
print(np.exp(x))
#print(math.exp(x))


# without broadcasting
a = np.array([1, 2, 3])
b = np.array([2, 2, 2])
print ("a * b is: ", a * b)


#broadcasting
x = x + 3 # more memory-efficient
print("x is: ", x)

# two dimensions are same if, they are equal or one of them is 1

# axes are compared starting from the trailing axis to upwards.

# dimensions with 1 are streched or 'copied' to match the other.


# broadcasting in practice
x = np.arange(4)
print("x shape: ", x.shape)

xx = x.reshape(4, 1)
y = np.ones(5)
z = np.ones((3,4))


#print ("x + y is: ", x + y)
print ("xx + y is: ", xx + y)



# outer-addition operation using broadcasting
a = np.array([0, 1, 2, 3])
b = np.array([1, 2, 3])
print("Curent a's shape is: ", a.shape)
a = a[:, np.newaxis] 
print("New a shape is: ", a.shape)


# implement sigmoid using numpy:
def numpy_sigmoid(x):
    s = 1 / (1 + np.exp(-1.0 * x))
    return s

x = np.array([1, 2, 3, 4])
print("Numpy's version of sigmoid:", numpy_sigmoid(x))


# sigmoid gradient, required for backpropogation implementation
# sigmoid graidient: sigma * (1- sigma)

def sigmoid_gradient(x):
    y = numpy_sigmoid(x) 
    g = numpy_sigmoid(x) * (1 - numpy_sigmoid(x))
    return g

print("Sigmoid graidnent is: ", sigmoid_gradient(x))


