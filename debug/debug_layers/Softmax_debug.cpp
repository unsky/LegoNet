#include <iostream>
#include <armadillo>
#include <../include/lego_net.hpp>

using namespace arma;
using namespace lego_net;
using std::shared_ptr;
namespace debug{

void testSoftmax() {
    //shared_ptr<Blob> x(new Blob(10,8,1,1,TRANDU));
    shared_ptr<Blob> x(new Blob(50,10,1,1,TRANDN));
    //(*x) *= 100;
    mat aa = randi<mat>(50, 1, distr_param(0,9));
    mat bb(50,10,fill::zeros);
    for (int i = 0; i < 50; ++i) {
        bb(i, (uword)aa(i,0)) = 1;
    }
    shared_ptr<Blob> y;
    mat2Blob(bb, y, x->size());
    vector<shared_ptr<Blob>> in{x, y};
    double loss;
    shared_ptr<Blob> out;
    shared_ptr<Blob> dummy_out;
    auto nfunc =[in, &dummy_out](double& e) {return SoftmaxLossLayer::go(in, e, dummy_out, 1);};
    SoftmaxLossLayer::go(in, loss, out);
    Blob num_dx = Test::calcNumGradientBlobLoss(in[0], nfunc);
    cout << "test softmax layer" << endl;
    cout << Test::relError(num_dx, *out) << endl;
}

}