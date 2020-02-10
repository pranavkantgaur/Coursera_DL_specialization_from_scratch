# Reimplementing the notebook: https://github.com/pranavkantgaur/CourseraDLSpecialization/blob/master/Course1/Week1/Python_Basics_With_Numpy_v3a.ipynb from scratch.


import math
import numpy as np
import time

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


# Reshaping arrays, exploring np.shape and np.reshape functions
# Note: np.reshape() and np.ndarray.reshape() are equivalent solutions to reshaping objects.
# np.reshape() is a function, whereas np.ndarray.reshape() is a method of class ndarray

x = np.zeros([256, 256, 3])
print("Current shape of x image is: ", x.shape)
x = x.reshape([256* 256 * 3, 1])
print("Image x shape is: ", x.shape)
y = x + 3 # result will be (256 X 256 X 3, 1)
print ("Shape of the augmented image is: ", y)



# implementing img2vect function for converting an 3-channel image to a 1D array
def img2vect(image):
  v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
  return v


image = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10,11,12],[13, 14, 15],[16, 17, 18]], [[19, 20, 21],[22, 23, 24],[25, 26, 27]]]) # 3 x3 x 3 array
v = img2vect(image)
print ("image vector: ", v)
print ("shape of image vector: ", v.shape)

# normalizing rows
def normalizeRows(x):
    x = np.linalg.norm(x, axis = 1, keepdims=True)
    return x


x = np.array([[1, 2, 3], [4, 5, 6]])
print("Original vector: ", x)
x_normalization_vector = normalizeRows(x)
normalized_x = x / x_normalization_vector # broadcasting, x.shape = (2, 3), x_normalization_vector.shape = (2, 1)
# x_normalization_vector reshaped into 2,3
print("Normalized vector: ", normalized_x)

# broadcasting and softmax function
def softmax(x):
    exp_x = np.exp(x)
    exp_x_normalization_vector = np.sum(exp_x, axis = 1, keepdims = True) #  keepdims=True makes sure that the following broadcasting will work.
    return exp_x / exp_x_normalization_vector

x = np.array([[1, 2, 3]]) # in nn forward inference, x will have shape (1, m), m: # training examples


x = np.array([
        [9, 2, 5, 0, 0],
            [7, 5, 0, 0 ,0]])

print("Softmax output: ", softmax(x))

# Vectorization: Classical dot, outer and elementwise product implementations
def classic_dot_product(v1, v2):
    vector_length = len(v1)
    dot_product = 0.0
    tick = time.process_time()
    for i in range(vector_length):
        dot_product += v1[i] * v2[i]
    tock = time.process_time()
    print("classic dot product took: ", tock - tick, "milliseconds!!")
    return dot_product        

v1 = [1, 2, 4, 5, 6] 
v2 = [7, 8, 9, 10, 11]
print("dot product of v1, v2: ", classic_dot_product(v1, v2))

def classic_outer_product(v1, v2):
    outer = np.zeros((len(v1), len(v2)))
    tick = time.process_time()
    for i in range(len(v1)):
        for j in range(len(v2)):
            outer[i, j] = v1[i] * v2[j]
    tock = time.process_time()
    print("classic outer product took: ", tock - tick, "milliseconds!!")
    return outer

print("outer product of v1, v2: ", classic_outer_product(v1, v2))


def classic_elementwise_product(v1, v2):
    elementwise_product = np.zeros((len(v1), 1))
    tick = time.process_time()
    for i in range(len(v1)):
        elementwise_product[i] = v1[i] * v2[i]
    tock = time.process_time()
    print("classic elementwise product took: ", tock-tick , "millisecods!!")
    return elementwise_product

print("classic elementwise product: ", classic_elementwise_product(v1, v2))


def classic_general_dot_product(v1):
    W = np.random.rand(3, len(v1))
    tick = time.process_time()
    gdot = np.zeros((W.shape[0], 1))
    for i in range(W.shape[0]):
        for j in range(len(v1)):
            gdot[i] += W[i, j] * v1[j]
    tock = time.process_time()
    print("classical general dot product took: ", tock - tick, "ms!!")
    return gdot 
   
print("classic general dot product: ", classic_general_dot_product(v1))

   
## Lets Vectorize this code!!
def vectorized_dot_product(v1, v2):
  tick = time.process_time()
  dot_product = np.dot(v1, v2)
  tock = time.process_time()
  print("vectorized dot product took: ", tock - tick, "ms!!")
  return dot_product

print("vectorized dot product: ", vectorized_dot_product(v1, v2))



def vectorized_outer_product(v1, v2):    
   tick = time.process_time()
   outer_product = np.outer(v1, v2)
   tock = time.process_time()
   print("vectorized outer product took: ", tock - tick, "ms!!")
   return outer_product

print("vectorized dot product: ", vectorized_outer_product(v1, v2))
  


def vectorized_elementwise_product(v1, v2):    
    tick = time.process_time()
    elementwise_prod = np.multiply(v1, v2)
    tock = time.process_time()
    print("vectorized elementwise product took: ", tock - tick, "ms!!")
    return elementwise_prod

print("vectorized elementwise product: ", vectorized_elementwise_product(v1, v2))


def vectorized_general_dot_product(v1):
    W = np.random.rand(3, len(v1))
    tick = time.process_time()
    general_dot_product = np.dot(W, v1)
    tock = time.process_time()
    print("vectorized general dot product took: ", tock - tick, "ms!!")
    return general_dot_product

print("vectorized general dot product: ", vectorized_general_dot_product(v1))

