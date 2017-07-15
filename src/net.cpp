
#include "../include/net.hpp"
#include <iostream>
using namespace std;

namespace lego_net {

void Net::trainNet(shared_ptr<Blob>& X,
                    shared_ptr<Blob>& Y,
                    NetParam& param,
                    std::string mode) {
    /*! fill X, Y */
    data_[layers_[0]][0] = X;
    data_[layers_.back()][1] = Y;

    // debug
    Blob pb, pd;

    /*! forward */
    int n = ltype_.size();
    for (int i = 0; i < n-1; ++i) {
        std::string ltype = ltype_[i];
        std::string lname = layers_[i];
        shared_ptr<Blob> out;
        if (ltype == "Conv") {
            int tF = param.params[lname].conv_kernels;
            int tC = data_[lname][0]->get_C();
            int tH = param.params[lname].conv_height;
            int tW = param.params[lname].conv_width;
            if (!data_[lname][1]) {
                data_[lname][1].reset(new Blob(tF, tC, tH, tW, TRANDN));
                (*data_[lname][1]) *= 1e-2;
            }
            if (!data_[lname][2]) {
                data_[lname][2].reset(new Blob(tF, 1, 1, 1, TRANDN));
                (*data_[lname][2]) *= 1e-1;
            }
            ConvLayer::forward(data_[lname], out, param.params[lname]);
        }
        if (ltype == "Pool") {
            PoolLayer::forward(data_[lname], out, param.params[lname]);
            pb = *data_[lname][0];
        }
        if (ltype == "Fc") {
            int tF = param.params[lname].fc_kernels;
            int tC = data_[lname][0]->get_C();
            int tH = data_[lname][0]->get_H();
            int tW = data_[lname][0]->get_W();
            if (!data_[lname][1]) {
                data_[lname][1].reset(new Blob(tF, tC, tH, tW, TRANDN));
                (*data_[lname][1]) *= 1e-2;
            }
            if (!data_[lname][2]) {
                data_[lname][2].reset(new Blob(tF, 1, 1, 1, TRANDN));
                (*data_[lname][2]) *= 1e-1;
            }
            AffineLayer::forward(data_[lname], out);
        }
        if (ltype == "Relu")
            ReluLayer::forward(data_[lname], out);
        if (ltype == "Dropout")
            DropoutLayer::forward(data_[lname], out, param.params[lname]);
        data_[layers_[i+1]][0] = out;
    }

    // calc loss
    std::string loss_type = ltype_.back();
    shared_ptr<Blob> dout;
    if (loss_type == "SVM")
        SVMLossLayer::go(data_[layers_.back()], loss_, dout);
    if (loss_type == "Softmax")
        SoftmaxLossLayer::go(data_[layers_.back()], loss_, dout);
    grads_[layers_.back()][0] = dout;

    loss_history_.push_back(loss_);

    if (mode == "forward")
        return;

    /*! backward */
    for (int i = n-2; i >= 0; --i) {
        std::string ltype = ltype_[i];
        std::string lname = layers_[i];
        if (ltype == "Conv")
            ConvLayer::backward(grads_[layers_[i+1]][0], data_[lname],
                                grads_[lname], param.params[lname]);
        if (ltype == "Pool") {
            PoolLayer::backward(grads_[layers_[i+1]][0], data_[lname],
                                grads_[lname], param.params[lname]);
        }
        if (ltype == "Fc")
            AffineLayer::backward(grads_[layers_[i+1]][0], data_[lname], grads_[lname]);
        if (ltype == "Relu")
            ReluLayer::backward(grads_[layers_[i+1]][0], data_[lname], grads_[lname]);
    }

    // regularition
    double reg_loss = 0;
    for (auto i : layers_) {
        if (grads_[i][1]) {
            // it's ok?
            Blob reg_data = param.reg * (*data_[i][1]);
            (*grads_[i][1]) = (*grads_[i][1]) + reg_data;
            reg_loss += data_[i][1]->sum();
        }
    }
    reg_loss *= param.reg * 0.5;
    loss_ = loss_ + reg_loss;

    return;
}

void Net::testNet(NetParam& param) {
    shared_ptr<Blob> X_batch(new Blob(X_train_->subBlob(0, 1)));
    shared_ptr<Blob> Y_batch(new Blob(Y_train_->subBlob(0, 1)));
    trainNet(X_batch, Y_batch, param);
    cout << "BEGIN TEST LAYERS" << endl;
    for (int i = 0; i < (int)layers_.size(); ++i) {
        testLayer(param, i);
        printf("\n");
    }
}

void Net::setup(NetParam& param,
                  vector<shared_ptr<Blob>>& X,
                  vector<shared_ptr<Blob>>& Y) {
    cout << "setup the network:" << endl;
    layers_ = param.layers;
    ltype_ = param.ltypes;

// setup data
    cout << "the input data: ";
    X[0]->shape_string();
    cout << "the label:";
    Y[0]->shape_string();
    data_[layers_[0]] = vector<shared_ptr<Blob>>(3);
    grads_[layers_[0]] = vector<shared_ptr<Blob>>(3);
    step_cache_[layers_[0]] = vector<shared_ptr<Blob>>(3);
    best_model_[layers_[0]] = vector<shared_ptr<Blob>>(3);

    data_[layers_[0]][0].reset(new Blob(X[0]->get_N(), X[0]->get_C(), X[0]->get_H(), X[0]->get_W(), TRANDN))
    // loss data_[0] ==> datas , data[1] ==>labels
    data_[layers_.back()][1].reset(new Blob(Y[0]->get_N(), Y[0]->get_C(), Y[0]->get_H(), Y[0]->get_W(), TRANDN));
    // debug
    Blob pb, pd;
    //use forward once to setup the network
    for (int i = 0; i < (int)layers_.size()-1; ++i) {
        cout << "creating " << lytype[i]<<": "<<endl;
        data_[layers_[i+1]] = vector<shared_ptr<Blob>>(3);
        grads_[layers_[i+1]] = vector<shared_ptr<Blob>>(3);
        step_cache_[layers_[i+1]] = vector<shared_ptr<Blob>>(3);
        best_model_[layers_[i+1]] = vector<shared_ptr<Blob>>(3);
        
        
        std::string ltype = ltype_[i];
        std::string lname = layers_[i];

        //bottom
        vector<shared_ptr<Blob>> bottom = data_[lname];

        cout<< "creating " << ltype << ": " << endl;
        cout<< "the bottom:";
        cout<< bottom[0]->shape_string();
        //up
        shared_ptr<Blob> up;
        if (ltype == "Conv") {
            int tF = param.params[lname].conv_kernels;
            int tC = bottom[0]->get_C();
            int tH = param.params[lname].conv_height;
            int tW = param.params[lname].conv_width;
            cout<< " the conv layer param: " << tF << "  " << tC << " " << tH << " " << tW <<endl;
            if (!bottom[1]) {
                bottom[1].reset(new Blob(tF, tC, tH, tW, TRANDN));
                (*bottom[1]) *= 1e-2;
            }
            if (!bottom[2]) {
                bottom[2].reset(new Blob(tF, 1, 1, 1, TRANDN));
                (*bottom[2]) *= 1e-1;
            }
            ConvLayer::forward(bottom, up , param.params[lname]);
        }
        if (ltype == "Pool") {
            PoolLayer::forward(bottom, up, param.params[lname]);
        }
        if (ltype == "Fc") {
            int tF = param.params[lname].fc_kernels;
            int tC = bottom[0]->get_C();
            int tH = bottom[0]->get_H();
            int tW = bottom[0]->get_W();
            if (!bottom[1]) {
                bottom[1].reset(new Blob(tF, tC, tH, tW, TRANDN));
                (*bottom[1]) *= 1e-2;
            }
            if (!bottom[2]) {
                bottom[2].reset(new Blob(tF, 1, 1, 1, TRANDN));
                (bottom[2]) *= 1e-1;
            }
            AffineLayer::forward(bottom, up);
        }
        if (ltype == "Relu")
            ReluLayer::forward(bottom, up);
        if (ltype == "Dropout")
            DropoutLayer::forward(bottom, up, param.params[lname]);
        cout << "the up :" << up->shape_string() << endl;
        data_[layers_[i+1]][0] = up;
    }

    // calc loss
    std::string loss_type = ltype_.back();
    cout<<"creating loss layer:"<<endl;
    cout<< loss_type<<"bottom: ";
    cout<< data_[layers_.back()][0]->shape_string();
    shared_ptr<Blob> dout;
    if (loss_type == "SVM")
        SVMLossLayer::go(data_[layers_.back()], loss_, dout);
    if (loss_type == "Softmax")
        SoftmaxLossLayer::go(data_[layers_.back()], loss_, dout);
    grads_[layers_.back()][0] = dout;
    loss_history_.push_back(loss_);
    cout<< "the loss output:" <<endl;
    cout << dout->shape_string();
    return;
}




void Net::testLayer(NetParam& param, int lnum) {
    std::string ltype = ltype_[lnum];
    std::string lname = layers_[lnum];
    if (ltype == "Fc")
        _test_fc_layer(data_[lname], grads_[lname], grads_[layers_[lnum+1]][0]);
    if (ltype == "Conv")
        _test_conv_layer(data_[lname], grads_[lname], grads_[layers_[lnum+1]][0], param.params[lname]);
    if (ltype == "Pool")
        _test_pool_layer(data_[lname], grads_[lname], grads_[layers_[lnum+1]][0], param.params[lname]);
    if (ltype == "Relu")
        _test_relu_layer(data_[lname], grads_[lname], grads_[layers_[lnum+1]][0]);
    if (ltype == "Dropout")
        _test_dropout_layer(data_[lname], grads_[lname], grads_[layers_[lnum+1]][0], param.params[lname]);
    if (ltype == "SVM")
        _test_svm_layer(data_[lname], grads_[lname][0]);
    if (ltype == "Softmax")
        _test_softmax_layer(data_[lname], grads_[lname][0]);
}

void Net::_test_fc_layer(vector<shared_ptr<Blob>>& in,
                         vector<shared_ptr<Blob>>& grads,
                         shared_ptr<Blob>& dout) {

    auto nfunc =[in](shared_ptr<Blob>& e) {return AffineLayer::forward(in, e); };
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    Blob num_dw = Test::calcNumGradientBlob(in[1], dout, nfunc);
    Blob num_db = Test::calcNumGradientBlob(in[2], dout, nfunc);

    cout << "Test Affine Layer:" << endl;
    cout << "Test num_dx and dX Layer:" << endl;
    cout << Test::relError(num_dx, *grads[0]) << endl;
    cout << "Test num_dw and dW Layer:" << endl;
    cout << Test::relError(num_dw, *grads[1]) << endl;
    cout << "Test num_db and db Layer:" << endl;
    cout << Test::relError(num_db, *grads[2]) << endl;

    return;
}

void Net::_test_conv_layer(vector<shared_ptr<Blob>>& in,
                     vector<shared_ptr<Blob>>& grads,
                     shared_ptr<Blob>& dout,
                     Param& param)  {

    auto nfunc =[in, &param](shared_ptr<Blob>& e) {return ConvLayer::forward(in, e, param); };
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    Blob num_dw = Test::calcNumGradientBlob(in[1], dout, nfunc);
    Blob num_db = Test::calcNumGradientBlob(in[2], dout, nfunc);

    cout << "Test Conv Layer:" << endl;
    cout << "Test num_dx and dX Layer:" << endl;
    cout << Test::relError(num_dx, *grads[0]) << endl;
    cout << "Test num_dw and dW Layer:" << endl;
    cout << Test::relError(num_dw, *grads[1]) << endl;
    cout << "Test num_db and db Layer:" << endl;
    cout << Test::relError(num_db, *grads[2]) << endl;

    return;
}
void Net::_test_pool_layer(vector<shared_ptr<Blob>>& in,
                     vector<shared_ptr<Blob>>& grads,
                     shared_ptr<Blob>& dout,
                     Param& param) {
    auto nfunc =[in, &param](shared_ptr<Blob>& e) {return PoolLayer::forward(in, e, param); };

    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);

    cout << "Test Pool Layer:" << endl;
    cout << Test::relError(num_dx, *grads[0]) << endl;

    return;
}

void Net::_test_relu_layer(vector<shared_ptr<Blob>>& in,
                     vector<shared_ptr<Blob>>& grads,
                     shared_ptr<Blob>& dout) {
    auto nfunc =[in](shared_ptr<Blob>& e) {return ReluLayer::forward(in, e); };
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);

    cout << "Test ReLU Layer:" << endl;
    cout << Test::relError(num_dx, *grads[0]) << endl;

    return;
}

void Net::_test_dropout_layer(vector<shared_ptr<Blob>>& in,
                     vector<shared_ptr<Blob>>& grads,
                     shared_ptr<Blob>& dout,
                     Param& param) {
    shared_ptr<Blob> dummy_out;
    auto nfunc =[in, &param](shared_ptr<Blob>& e) {return DropoutLayer::forward(in, e, param); };

    cout << "Test Dropout Layer:" << endl;
    Blob num_dx = Test::calcNumGradientBlob(in[0], dout, nfunc);
    cout << Test::relError(num_dx, *grads[0]) << endl;

    return;
}

void Net::_test_svm_layer(vector<shared_ptr<Blob>>& in,
                     shared_ptr<Blob>& dout) {
    shared_ptr<Blob> dummy_out;
    auto nfunc =[in, &dummy_out](double& e) {return SVMLossLayer::go(in, e, dummy_out, 1); };
    cout << "Test SVM Loss Layer:" << endl;

    Blob num_dx = Test::calcNumGradientBlobLoss(in[0], nfunc);
    cout << Test::relError(num_dx, *dout) << endl;

    return;
}

void Net::_test_softmax_layer(vector<shared_ptr<Blob>>& in,
                     shared_ptr<Blob>& dout) {
    shared_ptr<Blob> dummy_out;
    auto nfunc =[in, &dummy_out](double& e) {return SoftmaxLossLayer::go(in, e, dummy_out, 1); };

    cout << "Test Softmax Loss Layer:" << endl;
    Blob num_dx = Test::calcNumGradientBlobLoss(in[0], nfunc);
    cout << Test::relError(num_dx, *dout) << endl;

    return;
}


} //namespace lego_net
