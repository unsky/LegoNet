
#include <iostream>
#include <armadillo>
#include <../include/lego_net.hpp>

using namespace arma;
using namespace lego_net;
using std::shared_ptr;



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
}

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
}

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



void testSampleNetTrain() {
    NetParam param;
    param.batch_size = 30;
    param.lr = 0.1;
    // momentum=0.9, lr_decay=0.99, lr=0.05
    param.momentum = 0.9;
    param.num_epochs = 500;
    /*! when testing num_gradiets, reg must set to 0 */
    param.reg = 0;
    param.update = "momentum";
    param.use_batch = true;
    param.acc_frequence = 1;
    param.lr_decay = 0.99;

    //shared_ptr<Blob> X(new Blob(100, 2, 16, 16, TRANDN));
    shared_ptr<Blob> X(new Blob(100, 2, 8, 8, TRANDN));
    //(*X) *= 100;
    shared_ptr<Blob> Y;
    mat aa = randi<mat>(100, 1, distr_param(0, 9));
    mat bb(100, 10, fill::zeros);
    for (int i = 0; i < 100; ++i) {
        bb(i, (uword)aa(i, 0)) = 1;
    }
    mat2Blob(bb, Y, 10, 1, 1);
    param.layers.push_back("conv1");
    param.params["conv1"].conv_width = 3;
    param.params["conv1"].conv_height = 3;
    param.params["conv1"].conv_pad = 1;
    param.params["conv1"].conv_stride = 1;
    param.params["conv1"].conv_kernels = 5;
    param.layers.push_back("relu1");
    param.layers.push_back("pool1");
    param.params["pool1"].setPoolParam(2,2,2);
    //param.params["pool1"].pool_height = 2;
    //param.params["pool1"].pool_width = 2;
    //param.params["pool1"].pool_stride = 2;
    param.layers.push_back("fc1");
    param.params["fc1"].fc_kernels = 10;
    param.layers.push_back("softmax");
    param.ltypes.push_back("Conv");
    param.ltypes.push_back("Relu");
    param.ltypes.push_back("Pool");
    param.ltypes.push_back("Fc");
    param.ltypes.push_back("Softmax");

    Net inst;
    vector<shared_ptr<Blob>> data{X,X};
    vector<shared_ptr<Blob>> label{Y,Y};
    inst.initNet(param, data, label);
    inst.train(param);
}

int main()
{
    //testSharedptr();
    //testArma();
    //testBlob();
    //testAffineLayer();
    //testConvLayer();
    //testPoolLayer();
    //testRelu();
    //testDropout();
    //testSoftmax();
    //testSVM();
    //testTest();
    //int n = 3;
    //testNet();
    //while (n--) {
    //    mat a(5,5,fill::randn);
    //    a.print("a\n");
    //}
    //Blob b(4,2,2,2,TRANDN);
    //b.reshape().print();
    testSampleNetTrain();

    return 0;
}
