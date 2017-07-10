
#include <iostream>
#include <armadillo>
#include <../include/lego_net.hpp>
#include"../debug/debug_layers.hpp"

using namespace arma;
using namespace lego_net;
using std::shared_ptr;
namespace debug{
void testAffineLayer() {
    shared_ptr<Blob> a(new Blob(5,3,2,2,TONES));
    shared_ptr<Blob> b(new Blob(10,3,2,2,TONES));
    shared_ptr<Blob> c(new Blob(10,1,1,1,TONES));
    shared_ptr<Blob> dout(new Blob(5,10,1,1,TRANDN));
    vector<shared_ptr<Blob>> in{a, b, c};
    shared_ptr<Blob> out;
    vector<shared_ptr<Blob>> grads(3, shared_ptr<Blob>());
    AffineLayer::backward(dout, in, grads);
    auto nfunc =[in](shared_ptr<Blob>& e) {return AffineLayer::forward(in, e);};
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    Blob num_dw = Test::calcNumGradientBlob(in[1], dout, nfunc);
    Blob num_db = Test::calcNumGradientBlob(in[2], dout, nfunc);

    cout << "test affine layer" << endl;
    cout << Test::relError(num_dx, *grads[0]) << endl;
    cout << Test::relError(num_dw, *grads[1]) << endl;
    cout << Test::relError(num_db, *grads[2]) << endl;

    return;
}
}