import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.misc import imread
import glob
import h5py
from PIL import Image
import matplotlib.image as mpimg
import sys


def relu(z):
    a = np.maximum(0,z)
    assert(a.shape == z.shape)
    return a

def sigmoid(z):
    a = 1/(1+np.exp(-z))
    assert( a.shape == z.shape)
    return a

def initalize_parameters(layers_dim):
    """n_x is the dimension of input data, n_y is dimension of output dat,n_h size of hidden layers"""
    #np.random.seed(3)
    parameters = {} # dictionary used to store the parameters
    L = len(layers_dim)
    for i in range(L-2):
        parameters["W"+ str(i+1)] = np.random.randn(layers_dim[i+1], layers_dim[i]) # * np.sqrt(np.divide(2,layers_dim[i]))  # he initialization for relu
        parameters["b"+str(i+1)] = np.zeros((layers_dim[i+1], 1))
        assert (parameters["W" + str(i+1)].shape == (layers_dim[i+1], layers_dim[i]))
        assert (parameters["b" + str(i+1)].shape == (layers_dim[i+1], 1))
    parameters["W" + str(L-1)] = np.random.randn(layers_dim[L-1], layers_dim[L - 2])# * np.sqrt(np.divide(1, layers_dim[L - 2]))  #he initialization, to avoid vanishing and exploding gradieints
    parameters["b" + str(L-1)] = np.zeros((layers_dim[L-1], 1))
    assert (parameters["W" + str(L-1)].shape == (layers_dim[L-1], layers_dim[L - 2]))
    assert (parameters["b" + str(L-1)].shape == (layers_dim[L-1], 1))
    return parameters


def forward_linear(A, W, b):
    """ This function returns the linear combination: Z"""
    Z= np.dot(W,A)+b
    cache=(A, W, b)
    assert(Z.shape == (W.shape[0], A.shape[1]))
    return Z,cache

def forward_linear_activation(A_prev, W, b, activation):
    Z,linear_cache = forward_linear(A_prev, W, b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    if activation == "relu":
        A = relu(Z)
    activation_cache = Z
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache=(linear_cache, activation_cache)
    return A,cache

def L_layer_forward_activation(X,parameters):
    caches = []
    NumberLayers = len(parameters)//2
    A = X
    for i in range(1,NumberLayers):
        A_prev=A
        A, cache = forward_linear_activation(A_prev,parameters["W"+str(i)],parameters["b"+str(i)],'relu')
        caches.append(cache)
    AL,cache = forward_linear_activation(A, parameters["W"+str(NumberLayers)],parameters["b"+str(NumberLayers)], "sigmoid")
    caches.append(cache)
    assert(AL.shape==(1,X.shape[1]))
    return AL, caches


def backward_sigmoid(dA, activation_cache ):
    Z = activation_cache
    A = sigmoid(Z)
    gZ_prime = np.multiply(A,(1 - A))
    dZ = np.multiply(dA,gZ_prime)
    assert(dZ.shape == activation_cache.shape)
    assert(gZ_prime.shape == activation_cache.shape)
    return dZ, gZ_prime


def backward_relu(dA, activation_cache):
    gZ_prime = np.int64(activation_cache>0)
    dZ =  np.multiply(dA, np.int64(activation_cache > 0))
    assert(gZ_prime.shape == dA.shape)
    assert(dZ.shape == dA.shape )
    return dZ, gZ_prime

def backward_linear(dZ,lmbda, linear_cache):
    """This function will translate dz linearly from one layer to another. Here cache comes from def forward_linear(A,W,b):
    """
    A_prev, W, b = linear_cache
    m= A_prev.shape[1]           #number of training data
    dW = 1/m * np.dot(dZ,A_prev.T) + np.multiply(lmbda / m , W,dtype= np.float64)
    db = 1/m * np.sum(dZ,axis=1,dtype=np.float64,keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    assert(dA_prev.shape == A_prev.shape)
    return dW, db, dA_prev


def backward_activation(dA,cache,lmbda,activation):
    """This function estimates dZ from dA obtained using backward_linear"""
    linear_cache, activation_cache= cache             # cache for any given layers
    Z = activation_cache
    if activation == "sigmoid":
        dZ, g = backward_sigmoid(dA, Z)                #sigmoid derivative
        dW, db, dA_prev = backward_linear(dZ, lmbda, linear_cache)
    elif activation == "relu":
        dZ, g = backward_relu(dA, Z)
        dW, db, dA_prev = backward_linear(dZ, lmbda,linear_cache)
    else:
        print("Error!! enter the correct activation function")
       
    return dW, db, dA_prev

def L_backward_propagation(AL,Y,caches,lmbda):
    """ This part of the code does the full backward propagation"""
    grad={}
    L = len(caches)                                                             # Number of layers in the ANN
    Y= Y.reshape(AL.shape)
    ##################################################
    dAL = -(np.divide(Y, AL) - np.divide((1 - Y), (1 - AL)))
    cache = caches[L-1]
    dW, db, dA_prev = backward_activation(dAL,cache,lmbda,"sigmoid")
    #################################
    grad["dA"+str(L)] = dA_prev
    grad["dW" + str(L)] = dW
    grad["db" + str(L)] = db
    for i in reversed(range(L-1)):
        cache=caches[i]
        dW, db, dA_prev = backward_activation(grad["dA"+str(i+2)],cache,lmbda,"relu")
        grad["dA"+str(i+1)] = dA_prev
        grad["dW" + str(i+1)] = dW
        grad["db" + str(i+1)] = db    

    return grad
#####################################################################################
##########################################################################################
def parameter_update(grad,learning_rate,parameters):
    """This function updates the paramters"""    
    L = len(parameters) //2
    for i in range(L):
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - learning_rate * grad["dW" + str(i+1)]
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - learning_rate * grad["db" + str(i+1)]
    return parameters
############################################################################################
def initialize_momentum(parameters):
    """ Initializes the first momentum"""
    momentum={}
    L= len(parameters) //2
    for i in range(L):
        momentum["dW" +str(i+1)] = np.zeros(parameters["W" + str(i+1)].shape)
        momentum["db" + str(i+1)] = np.zeros(parameters["b" + str(i+1)].shape)
        ###############################################################################
        assert(momentum["dW" +str(i+1)].shape == parameters["W" +str(i+1)].shape)
        assert(momentum["db" +str(i+1)].shape == parameters["b" +str(i+1)].shape)
    return momentum
########################################################################################
def initialize_s_momentum(parameters):
    """Initializes the second momentum of weights"""
    s_momentum={}
    L= len(parameters)//2
    for i in range(L):
        s_momentum["dW" +str(i+1)] = np.zeros(parameters["W" + str(i+1)].shape)
        s_momentum["db" +str(i+1)] = np.zeros(parameters["b" + str(i+1)].shape)
        assert(s_momentum["dW" +str(i+1)].shape == parameters["W" +str(i+1)].shape)
        assert(s_momentum["db" +str(i+1)].shape == parameters["b" +str(i+1)].shape)
    return s_momentum
    
def parameter_update_momentum(parameters,grad, momentum, learning_rate,beta):
    """This optimization algorithm uses Gradient descent with momentum to optimize the parameters"""    
    L = len(grad) // 3  # 3 because grad consists of dA, dW , db
    for i in range(L):
        momentum["dW" +str(i+1)] = beta * momentum["dW" +str(i+1)] + (1-beta) * grad["dW" + str(i+1)]
        momentum["db" +str(i+1)] = beta * momentum["db" +str(i+1)] + (1-beta) * grad["db" + str(i+1)]
        
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - learning_rate * momentum["dW" +str(i+1)]
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - learning_rate * momentum["db" +str(i+1)]
        
    return parameters, momentum
    
def RMS_prop_update(parameters, grad, momentum, learning_rate, beta):
    epsilon = np.float64(1e-8)
    L= len(grad) // 3
    for i in range(L):
        s_momentum["dW" + str(i+1)] = beta * s_momentum["dW" + str(i+1)] + (1-beta) * np.square(grad["dW" +str(i + 1)])
        s_momentum["db" + str(i+1)] = beta * s_momentum["db" + str(i+1)] + (1-beta) * np.square(grad["db" +str(i + 1)])
        
        parameters["W" +str(i+1)] = parameters["W" +str(i+1)] - learning_rate * np.divide(grad["dW" + str(i+1)], (np.sqrt(s_momentum["dW" +str(i+1)]) +epsilon))
        parameters["b" +str(i+1)] = parameters["b" +str(i+1)] - learning_rate * np.divide(grad["db" + str(i+1)], (np.sqrt(s_momentum["db" +str(i+1)]) +epsilon))
    
    return parameters, s_momentum       
        
    
    
def adam_algorithm(parameters, grad, momentum,s_momentum,learning_rate, beta1, beta2,t):
    "beta1, beta2 are the exponential decay rates for moment estimates"
    L= len(grad) // 3
    epsilon = 1e-8
    
    for i in range(L):
        momentum["dW"] = beta1 * momentum["dW" + str(i+1)] + (1-beta1) * grad["dW" + str(i+1)]
        momentum["db"] = beta1 * momentum["db" + str(i+1)] + (1-beta1) * grad["db" + str(i+1)]        
        momentum["dW"] = np.divide(momentum["dW"] , (1- beta1**t))
        momentum["db"] = np.divide(momentum["db"] , (1- beta1**t))
        
        s_momentum["dW" +str(i+1)] = beta2 * s_momentum["dW" + str(i+1)] + (1- beta2) * np.square(grad["dW" + str(i+1)])
        s_momentum["db" +str(i+1)] = beta2 * s_momentum["db" + str(i+1)] + (1- beta2) * np.square(grad["db" + str(i+1)])
        s_momentum["dW" +str(i+1)] = np.divide(s_momentum["dW"+str(i+1)] , (1- beta2**t))
        s_momentum["db" +str(i+1)] = np.divide(s_momentum["db"+str(i+1)] , (1- beta2**t))

        learning_rate_t = learning_rate * np.divide(np.sqrt(1- beta2 ** t), (1- beta1**t))                            # from Adams paper, increases efficiency   
        
        parameters["W" +str(i+1)] = parameters["W" +str(i+1)] - learning_rate_t * np.divide(momentum["dW" + str(i+1)], (np.sqrt(s_momentum["dW" +str(i+1)]) +epsilon))
        parameters["b" +str(i+1)] = parameters["b" +str(i+1)] - learning_rate_t * np.divide(momentum["db" + str(i+1)], (np.sqrt(s_momentum["db" +str(i+1)]) +epsilon))
    return parameters, momentum, s_momentum  

    
    
"""
def Load_data(path):    
    filelist = glob.glob(path+"/*.ppm")
    #x = np.array([np.array(Image.open(fname)) for fname in filelist])
   #    imag_list=np.zeros(110)
    for filename in glob.glob(path+"/*.ppm"):
        img = imread(filename,mode='RGB')
        temp_img = img.reshape(img.shape[0]*img.shape[1]*img.shape[2],1)
        image_list.append(temp_img)

    return image_list
"""

def predict(X, parameters):
    AL,_ = L_layer_forward_activation(X,parameters)
    Y_prediction = np.zeros(AL.shape)
    for i in range(AL.shape[1]):
        if AL[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    assert (Y_prediction.shape == AL.shape)
    return Y_prediction


def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def cost_function(AL, Y ):
    """Unregularized cost function"""
    m =  Y.shape[1]
    cost = -1/m * (np.dot(Y,np.log(AL.T)) + np.dot((1-Y),np.log(1-AL.T)))
    cost = np.squeeze(cost)
    assert(cost.shape ==())
    return cost
def compute_cost_regularization(AL,Y,parameters,lmbda):
    """Returns regularized cost function"""
    m= Y.shape[1]
    cost = -1.0 / m * (np.dot(Y,np.log(AL.T)) + np.dot((1-Y),np.log(1-AL.T)))
    L = len(parameters)//2
    regularization_value = 0
    for i in range(L):
        regularization_value = regularization_value + np.sum(np.square(parameters["W" +str(i+1)]),dtype= np.float64)
    cost_regularization = cost + lmbda / (2*m) * regularization_value
    cost_regularization= np.squeeze(cost_regularization)
    assert(cost_regularization.shape == ())
    return cost_regularization

def Gradient_check(X,Y, parameters, gradients, epsilon,layers_dims,lmbda):
   # parameters: Estimated set of parameters estimated using Back-Propagation
   # gradients: estimated using back-propagation
   # layer_dims: This is the list containing number of nodes in different hidden layers      
    vectorized_param = dictionary_to_vector(parameters)
    grads = gradient_to_vector(gradients)           
    num_parameters = vectorized_param.shape[0]
    J_plus = np.zeros((num_parameters,1))
    J_minus = np.zeros((num_parameters,1))
    GraDderive = np.zeros((num_parameters,1))   
    for i in range(num_parameters):
        print(i)
        vectorized_param_plus = np.copy(vectorized_param)
        vectorized_param_minus = np.copy(vectorized_param)
        vectorized_param_plus[i][0] = vectorized_param_plus[i][0]+ epsilon
        vectorized_param_minus[i][0] = vectorized_param_minus[i][0]- epsilon
        param_plus = vector_to_dictionary(vectorized_param_plus,layers_dims)
        param_minus = vector_to_dictionary(vectorized_param_minus,layers_dims)
        y_hat_plus, _= L_layer_forward_activation(X,param_plus)
        y_hat_minus, _= L_layer_forward_activation(X,param_minus)
             
        #J_plus[i] = cost_function(y_hat_plus, Y )
        #J_minus[i] = cost_function(y_hat_minus, Y )
        J_plus[i]  =  compute_cost_regularization(y_hat_plus,Y,param_plus,lmbda)
        J_minus[i] = compute_cost_regularization(y_hat_minus,Y,param_minus,lmbda)
        GraDderive[i] = np.divide((J_plus[i] - J_minus[i]), (2 * epsilon), dtype = np.float64)
        
    numerator = np.linalg.norm(GraDderive-grads)
    denominator = np.linalg.norm(grads) + np.linalg.norm(GraDderive)
    error =  np.divide(numerator, denominator, dtype = np.float64)
    
    if error > epsilon:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(error) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(error) + "\033[0m")
    
    return error, GraDderive
 
def vector_to_dictionary(vector,layer_dims):
 #   Rhis function will convert vector into paramters
    L = len(layer_dims)
    init=0
    param={}
    for i in range(L-1):
        coeff_len = layer_dims[i + 1] * layer_dims[i]
        vec_len = init + coeff_len
        param["W" + str(i+1)] = vector[init : vec_len,0].reshape((layer_dims[i+1],layer_dims[i]))
        bias_len = vec_len+ layer_dims[i+1]
        param["b" + str(i+1)] = vector[vec_len:bias_len , 0].reshape((layer_dims[i+1],1))
        init = bias_len
    return param


def dictionary_to_vector(parameters):
#    This function will convert parameters dictionary to vectors
    L=len(parameters) // 2
    count=0
    #parameter_vector=[]
    
    for i in range(L):
        tempW = np.reshape(parameters["W" + str(i + 1)],(-1,1))
        tempb = np.reshape(parameters["b" + str(i + 1)],(-1,1))
        temp_theta = np.concatenate((tempW,tempb),axis=0)
        if count == 0:
            theta = temp_theta           
        else:
            theta = np.concatenate((theta,temp_theta), axis=0)
        count = count + 1
             
    return theta 

def gradient_to_vector(grads):
 #   This function will convert parameters dictionary to vectors
    L=len(grads) // 3
    count=0
    for i in range(L):
        tempW = np.reshape(grads["dW" + str(i + 1)],(-1,1))
        tempb = np.reshape(grads["db" + str(i + 1)],(-1,1))
        temp_theta = np.concatenate((tempW,tempb),axis=0)
        if count == 0:
            theta = temp_theta           
        else:
            theta = np.concatenate((theta,temp_theta), axis=0)
        count = count + 1

    return theta
   
def mini_batch(train_x, train_y,mini_batch_size):
    m = train_x.shape[1]
    num_complete_batch = math.floor(m / mini_batch_size)
    permutation = list(np.random.permutation(m))
    ShuffleX= train_x[:,permutation]
    ShuffleY= train_y[:,permutation]
    miniBatches=[]   
    for i in range(num_complete_batch):
        batchX = ShuffleX[:, (i * mini_batch_size):((i+1) * mini_batch_size)]
        batchY = ShuffleY[:,(i * mini_batch_size):((i+1) * mini_batch_size)]
        
        miniBatch =(batchX, batchY)
        miniBatches.append(miniBatch)
    if m % mini_batch_size !=0:
        batchX= ShuffleX[:,(m - m % mini_batch_size):m]
        batchY= ShuffleY[:,(m - m % mini_batch_size):m]
    miniBatch =(batchX, batchY)
    miniBatches.append(miniBatch)
    return miniBatches
  
def batch_normalization_forward(x, beta, gamma, epsilon):
    n, d = x.shape
    mu = np.mean(x, axis=1)
    sig = np.sqrt(np.var(x,axis =1))
    stand_x = (x - mu) / (sig + epsilon)
    beta_x = stand_x * beta
    gamma_x = beta_x + gamma
    return gamma_x

def prediction_error(y_hat,y):
    """y_hat is the prediction, y is the actual observation"""    
    y_hat = y_hat.reshape((y.shape[0],y.shape[1]))      
    error = np.sum(np.abs(y_hat-y))
    error_fraction = np.divide(error, y.shape[1]) * 100
    return error_fraction    

def layer_configuration(n_x,number_h,hidden_dimension,n_y):
    """The dimension of hidden layer"""
    layer_dim=[]
    layer_dim.append(n_x)
    for i in range(number_h):
        layer_dim.append(hidden_dimension[i])
    layer_dim.append(n_y)
    return layer_dim
    

#np.random.seed(1)
train_X, train_y, test_set_x_orig, test_set_y_orig, classes = load_dataset()
cost_list=[]
train_X_flatten = train_X.reshape(train_X.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
test_set_x_flatten= test_set_x_flatten/ 255
train_X_flatten = train_X_flatten / 255
#learning_rate = 0.007
GradienntCheck= True
lr0= 0.001                                                                     # initial learning rate
drop =0.5
epochs_drop = 10                                                               # drop the learning rate by half every 10 epochs
n_x = train_X_flatten.shape[0]                                                 # num_px * num_px * 3
n_hid_layer = 2
hidden_dimension= [2,2]
n_y = 1

layers_dims = layer_configuration(n_x,n_hid_layer,hidden_dimension,n_y)        #[n_x, n_h1, n_h2, n_h3, n_h4, n_h5, n_y]
number_of_iteration = 1000
epsilon= 1e-7                                                                  #grad check parameter
parameters = initalize_parameters(layers_dims)
s_momentum = initialize_s_momentum(parameters)
momentum = initialize_momentum(parameters)
algorithm = "1"
mini_batch_size = 128
learning_rate =lr0
lmbda =0.3
for epoch in range(number_of_iteration):    
    batch_data=mini_batch(train_X_flatten, train_y,mini_batch_size)
    for batch in batch_data:
        (batch_x , batch_y) = batch
        AL, caches = L_layer_forward_activation(batch_x, parameters)
        #cost_old = cost_function(AL,batch_y)
        cost =  compute_cost_regularization(AL,batch_y,parameters,lmbda)
        grad = L_backward_propagation(AL, batch_y,caches,lmbda)
        if algorithm == "Grad_momentum":
            parameters,_ = parameter_update_momentum(parameters,grad, momentum, learning_rate,0.9)        
        elif algorithm=="RMSprop":
            parameters,_ = RMS_prop_update(parameters, grad, momentum, learning_rate, 0.9)
        elif algorithm == "adam":
            parameters,_,_ = adam_algorithm(parameters, grad, momentum, s_momentum,learning_rate, 0.9, 0.999,(epoch + 1))        
        else:
            parameters = parameter_update(grad, learning_rate, parameters)    
    
        #if epoch==10:
            #difference, gradApprox = Gradient_check(batch_x, batch_y, parameters, grad, epsilon,layers_dims)
         #   difference, gradApprox = Gradient_check(batch_x, batch_y, parameters, grad, epsilon,layers_dims, lmbda)
          #  sys.exit()
          #  break
        
        cost_list.append(cost)
    #learning_rate = np.power(0.95, epoch) * learning_rate                                 #exponentially decaying learning rate
    learning_rate = lr0 * drop**math.floor(epoch / epochs_drop) 
    if(epoch % 100==0):
        print(" The cost of network at epoch: % d/%d is  : %f " %(epoch, number_of_iteration,cost))

##########################
perm = list(np.random.permutation(test_set_x_flatten.shape[1]))
dev_perm = perm[0:len(perm)//2]
test_perm =list(set(perm)- set(dev_perm))
##################################
dev_x = test_set_x_flatten[:,dev_perm]
test_x = test_set_x_flatten[:,test_perm]
dev_y =  test_set_y_orig[:,dev_perm]
test_y = test_set_y_orig[:,test_perm]
########################################
y_hat_training = predict(train_X_flatten,parameters)
y_hat_dev = predict(dev_x,parameters)
y_hat_test = predict(test_x,parameters)
 ####################################################
trainin_error_rate = prediction_error(y_hat_training,train_y)
dev_error_rate = prediction_error(y_hat_dev,dev_y)
test_error_rate = prediction_error(y_hat_test,test_y)

print("The training set error on the data is:"+ str(trainin_error_rate))
print("The dev set error is:"+ str(dev_error_rate))
print("The test set error is:" + str(test_error_rate))

plt.figure()
plt.plot(cost_list)


























