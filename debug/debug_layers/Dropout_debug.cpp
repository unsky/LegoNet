
#include <iostream>
#include <armadillo>
#include <../include/lego_net.hpp>

using namespace arma;
using namespace lego_net;
using std::shared_ptr;
namespace debug{
    void testDropout() {
    shared_ptr<Blob> x(new Blob(1,1,5,5,TRANDN));
    shared_ptr<Blob> dout(new Blob(1,1,5,5,TRANDN));
    vector<shared_ptr<Blob>> in{x};
    shared_ptr<Blob> out;
    Param param;
    param.setDropoutpParam(3, 0.5, 123);
    vector<shared_ptr<Blob>> grads(3, shared_ptr<Blob>());
    shared_ptr<Blob> dummy_out;
    auto nfunc =[in, &param](shared_ptr<Blob>& e) {return DropoutLayer::forward(in, e, param);};
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    DropoutLayer::backward(dout, in, grads, param);
    cout << "test dropout layer" << endl;
    cout << Test::relError(num_dx, *grads[0]) << endl;
}}