int main() {
    vector<double> weights1{1,2,3};
    vector<double> weights2{1,2.5,3.5};
    vector<double> weights3{4,1,2.5};

    vector<double> input{1,2,3};

    double bias1 = 1;
    double bias2 = 2;
    double bias3 = 3;

    double res = (weights1[0] * input[0] + weights1[1] * input[1] + weights1[2] * input[2]) + bias1 +
                (weights2[0] * input[0] + weights2[1] * input[1] + weights2[2] * input[2]) + bias2 +
                (weights3[0] * input[0] + weights3[1] * input[1] + weights3[2] * input[2]) + bias3;
    return res;
}