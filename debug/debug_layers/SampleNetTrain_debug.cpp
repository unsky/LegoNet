#include <iostream>
#include <armadillo>
#include <../include/lego_net.hpp>

using namespace arma;
using namespace lego_net;
using std::shared_ptr;
namespace debug{

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
}// debug

