
#include <iostream>
#include <armadillo>
#include <../include/lego_net.hpp>

using namespace arma;
using namespace lego_net;
using std::shared_ptr;
namespace debug{
    void testRelu() {
    shared_ptr<Blob> x(new Blob(1,1,5,5,TRANDN));
    shared_ptr<Blob> dout(new Blob(1,1,5,5,TRANDN));
    vector<shared_ptr<Blob>> in{x};
    shared_ptr<Blob> out;
    vector<shared_ptr<Blob>> grads(3, shared_ptr<Blob>());
    ReluLayer::backward(dout, in, grads);
    auto nfunc =[in](shared_ptr<Blob>& e) {return ReluLayer::forward(in, e);};
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    cout << "test relu layer" << endl;
    cout << Test::relError(num_dx, *grads[0]) << endl;
}
}