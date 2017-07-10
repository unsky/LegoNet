#include <iostream>
#include <armadillo>
#include <../include/lego_net.hpp>

using namespace arma;
using namespace lego_net;
using std::shared_ptr;
namespace debug{


    void testSVM() {
    shared_ptr<Blob> x(new Blob(10, 8, 1, 1, TRANDU));
    mat aa = randi<mat>(10, 1, distr_param(0, 7));
    mat bb(10, 8, fill::zeros);
    for (int i = 0; i < 10; ++i) {
        bb(i, (uword)aa(i, 0)) = 1;
    }
    shared_ptr<Blob> y;
    mat2Blob(bb, y, x->size());
    vector<shared_ptr<Blob>> in{x, y};
    double loss;
    shared_ptr<Blob> out;
    SVMLossLayer::go(in, loss, out);
    shared_ptr<Blob> dummy_out;
    auto nfunc =[in, &dummy_out](double& e) {return SVMLossLayer::go(in, e, dummy_out, 1);};
    Blob num_dx = Test::calcNumGradientBlobLoss(in[0], nfunc);
    cout << "test svm layer" << endl;
    cout << Test::relError(num_dx, *out) << endl;
}
}