#ifndef DEF_NETWORK
#define DEF_NETWORK

#include <vector> //Ne pas oublier !

#include "Matrix.h"
using namespace std;


class Network
{
    public:
        Network(std::vector<int>sizes , int epochs, int mini_batch_size, int eta);
        Network(const char *filepath); //a quoi ça sert?

        std::vector<Matrix <double>> weights;
        std::vector<Matrix <double>> biases;

        void evaluate(std::vector<Matrix < double>> data_to_test);
        Matrix<double> feedforward(std::vector<double> input);

        static double normal_random(double x);
        static double sigmoid(double x);
        static double sigmoid_prime(double x);
        static double ReLU(double x);
        static double ReLU_prime(double x);

};


#endif // NETWORK_H_INCLUDED
