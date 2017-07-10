
#include <iostream>
#include <armadillo>
#include <../include/lego_net.hpp>

using namespace arma;
using namespace lego_net;
using std::shared_ptr;
namespace debug{
    void testConvLayer() {
    shared_ptr<Blob> x(new Blob(1,3,5,5,TRANDN));
    shared_ptr<Blob> w(new Blob(2,3,3,3,TRANDN));
    shared_ptr<Blob> b(new Blob(2,1,1,1,TRANDN));
    shared_ptr<Blob> dout(new Blob(1,2,5,5,TRANDN));
    Param param;
    param.setConvParam(1,1,3,3,5);
    vector<shared_ptr<Blob>> in{x, w, b};
    vector<shared_ptr<Blob>> grads(3, shared_ptr<Blob>());
    grads.push_back(shared_ptr<Blob>());
    grads.push_back(shared_ptr<Blob>());
    grads.push_back(shared_ptr<Blob>());
    ConvLayer::backward(dout, in, grads, param);

    /*! test num_grads */
    auto nfunc =[in, &param](shared_ptr<Blob>& e) {return ConvLayer::forward(in, e, param);};
    cout << "test conv layer" << endl;
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    Blob num_dw = Test::calcNumGradientBlob(in[1], dout, nfunc);
    Blob num_db = Test::calcNumGradientBlob(in[2], dout, nfunc);

    cout << Test::relError(num_dx, *grads[0]) << endl;
    cout << Test::relError(num_dw, *grads[1]) << endl;
    cout << Test::relError(num_db, *grads[2]) << endl;

    return;
}
}