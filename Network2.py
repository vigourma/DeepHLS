import random
import numpy as np
import mnist
import time
import pickle 


class Network(object):

    def __init__(self, sizes, epochs, mini_batch_size, eta):
        #hyperparameters (to optimize ---> best 96.2) nn = Network([784,60,10], 20, 5, 3.0)
        # nn = Network([784,100,10], 100, 10, 3.0)  -->96.59 %
        # nn = Network([784,150,10], 150, 10, 3.0)  -->97%
        self.num_layers = len(sizes) #number of layers
        
        self.sizes = sizes #[number of input pixels, number of neuron, number of digit]
        
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
                       
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.epochs = epochs # Number of Epochs through which Neural Network will be trained
        
        self.mini_batch_size = mini_batch_size #Size of the mini batch of training examples as used by Stochastic Gradient Descent
        
        self.eta = eta #learning rate for the gradient descent optimization
        
        
        #instrumentation
        self.training_time = 0 #number of epochs done
        #time of calculation (removed to improve time calculation)
        self.Tfeedforward = [0]
        self.Tbackforward = [0]
        self.Tthird = [0]
        self.TminiBatchCreation =[0]
        
        
        
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

        
        
        
        
    def backprop(self, x, y):
        #initialization of vectors
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
                   
        # feedforward
        #t0 = time.time()
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z) #store all q vectors
            activation = sigmoid(z) 
            activations.append(activation) #store all Q vectors
        #t1 = time.time()-t0
        #self.Tfeedforward.append(t1)
            
        # backward pass
        #t0 = time.time()
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) #calculation of delta1
        nabla_b[-1] = delta #delta nablaB 1
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #delta nablaW 1

        #loop for only 1 time but can be used to extend to add more layers
        for i in range(2, self.num_layers):
            z = zs[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * sp #calculation of delta0
            nabla_b[-i] = delta #delta nablaB 0
            nabla_w[-i] = np.dot(delta, activations[-i-1].transpose()) #delta nablaW 0
        #t1 = time.time()-t0
        #self.Tbackforward.append(t1)
        return (nabla_b, nabla_w)   #size 2 and 2     





        
    def update_mini_batch(self, mini_batch, eta):
        #initialisation of vectors
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #calculation of nablaB and nablaW
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) 
            #t0 = time.time()
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] #size 2 , calculation of nablaB
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #size 2 , calculation of nablaW
            #self.Tthird.append(time.time()-t0)
        #t0 = time.time()
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] #update of coef of W
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)] #up^date of coef of B
        #self.Tthird.append(time.time()-t0)   
        




    
    def SGD(self, training_data):
        n = len(training_data)
        #loop on the number of training to do
        for j in range(self.epochs):
            #t0 = time.time()
            self.training_time += 1 
            random.shuffle(training_data) #randomize learning 
            mini_batches = [ training_data[k : k+ self.mini_batch_size] for k in range(0, n, self.mini_batch_size)] #create mini batches
            #self.TminiBatchCreation.append(time.time()-t0)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, self.eta) #learning on each mini batch


                
                
        
        
    def evaluate(self, data_to_test):
        #check the result on each couple used for test and return % of correct answer
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data_to_test]
        return sum(int(x == y) for (x, y) in test_results)

        
        
              
        
    def cost_derivative(self, output_activations, y):
        #cost function used
        return (output_activations-y)

        
        
        
        
#### Miscellaneous functions
def sigmoid(x):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-x))    
    
def sigmoid_prime(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x)*(1-sigmoid(x))
 
    
def ReLU(x):
    return x * (x > 0)

def ReLU_prime(x):
    return 1. * (x > 0)    
    

    
    
    
#saving network   
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)  
        
        
        
        
        
# function to select depending of objective        
def verbose_time_calculation_repartition():
    print ("--------------- Loading data ---------------")
    t0 = time.time()
    training_data, validation_data, test_data = mnist.load_data()
    print ("--------------- Data loaded ---------------")
    Tload = (time.time()-t0)
    print("Time to load picture : " , Tload)
    
    nn = Network([784,60,10], 20, 5, 3.0)
    print ("--------------- Training ---------------")
    t0 = time.time()
    nn.SGD(training_data)
    Ttrain = (time.time()-t0)
    print ("--------------- Trained ---------------")
    print("Time to train model : " , Ttrain)
    
    t0 = time.time()
    print ("Neural Network accuracy on test data is {} %".format(nn.evaluate(test_data) / 100.00))
    Ttest = (time.time()-t0)
    print("Time to test : " , Ttest)
    print("----------------- More Info --------------")
    print ("Time in feedforward for training", np.sum(nn.Tfeedforward))
    print ("Time in backforward for training", np.sum(nn.Tbackforward))
    print ("Time in third part  for training", np.sum(nn.Tthird))
    print ("Time to create minibatches " , np.sum(nn.TminiBatchCreation))
    print ("----------------- Repartition -----------")
    print ("Tfeedforward : ", np.sum(nn.Tfeedforward) / Ttrain * 100, "%")
    print ("Tbackforward : ", np.sum(nn.Tbackforward) / Ttrain * 100, "%")
    print ("Tthird : ", np.sum(nn.Tthird) / Ttrain * 100, "%")
    print ("TminiBatchCreation : ", np.sum(nn.TminiBatchCreation) / Ttrain * 100, "%")
    print ("--------------------- THE END -----------------")    
    
if __name__ == '__main__':
    print ("--------------- Loading data ---------------")
    training_data, validation_data, test_data = mnist.load_data()
    print ("--------------- Data loaded ---------------")
    nn = Network([784,60,10], 20, 5, 3.0)
    print ("--------------- Training ---------------")
    nn.SGD(training_data)
    print ("--------------- Trained ---------------")
    print ("Neural Network accuracy on test data is {} %".format(nn.evaluate(test_data) / 100.00))
    print ("--------------------- THE END -----------------")
