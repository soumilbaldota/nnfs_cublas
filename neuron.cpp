int main() {
    vector<vector<double>> weights{{1,2,3}, {1,2.5,3.5}, {4,1,2.5}};
    vector<double> input{1,2,3};
    double bias = {3,2,1};

    for(int _w = 0; _w < weights.size(); _w++) {
        double res = bias[_w];
        for(int _i = 0; _i < input.size(); _i++) {
            res += input[_i] * weights[_w][_i];
        }
    }
    
    return res;
}