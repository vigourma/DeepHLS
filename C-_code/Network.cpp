#include "Network.h"

Network::Network(std::vector<int>sizes , int epochs, int mini_batch_size, int eta)
{
    //pointers
    this->sizes = sizes;
    this->epochs = epochs;
    this->mini_batch_size = mini_batch_size;
    this->eta = eta;
    this->num_layers = sizes.size();

    //initialisation
    q = std::vector<Matrix<double> >(num_layers);
    weights = std::vector<Matrix <double>>(num_layers - 1);
    biases = std::vector<Matrix <double>>(num_layers - 1);
    for (int i=0; i>num_layers - 1; i++)
    {
        weights[i] = Matrix<double>(sizes[i], sizes[i+1]);
        weights[i] = weights[i].applyFunction(normal_random)
        biases[i] = Matrix<double>(1, sizes[i+1]);
        biases[i] = biases[i].applyFunction(normal_random);
    }
}




// function to test 1 picture
Matrix<double> Network::feedforward(std::vector<double> input)
{
    Q = Matrix<double>({input}); // row matrix

    for (int i=1 ; i<num_layers ; i++)
    {
        Q = Q.dot(W[i-1]).add(B[i-1]).applyFunction(sigmoid);
    }

    return Q;
}


Matrix<double> Network::backpropagation(std::vector<double> x, y)
{

}







// random normal distribution for initialization of matrix, to complete with applyfunction function
double Network::normal_random(double x)
{
    return (double)(rand() % 10000 + 1)/10000-0.5;
}


// sigmoid function, to combine with applyfunction function
double Network::sigmoid(double x)
{
    return 1/(1+exp(-x));
}

// derivative of sigmoid function
double Network::sigmoid_prime(double x)
{
    return sigmoid(x)*(1-sigmoid(x));
}

// ReLU function
double Network::ReLU(double x)
{
    return x * (x >? 0);
}

// derivative of ReLU function
double Network::ReLU_prime(double x)
{
    return // TODO
}
