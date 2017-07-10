
#include <iostream>
#include <armadillo>
#include <../include/lego_net.hpp>

using namespace arma;
using namespace lego_net;
using std::shared_ptr;
namespace debug{
void testPoolLayer() {
    shared_ptr<Blob> x(new Blob(1,5,4,4,TRANDN));
    //shared_ptr<Blob> x(new Blob(1,5,4,4,TZEROS));
    (*x) *= 1e-2;
    shared_ptr<Blob> dout(new Blob(1,5,2,2,TRANDN));
    vector<shared_ptr<Blob>> in{x};
    shared_ptr<Blob> out;
    Param param;
    param.setPoolParam(2,2,2);
    PoolLayer::forward(in, out, param);
    vector<shared_ptr<Blob>> grads(3, shared_ptr<Blob>());
    PoolLayer::backward(dout, in, grads, param);
    auto nfunc =[in, &param](shared_ptr<Blob>& e) {return PoolLayer::forward(in, e, param);};
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    cout << "test pool layer" << endl;
    cout << Test::relError(num_dx, *grads[0]) << endl;
}}