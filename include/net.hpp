

#ifndef LEGO_NET_NET_HPP_
#define LEGO_NET_NET_HPP_

#include "blob.hpp"
#include "layer.hpp"
#include "test.hpp"
#include <json/json.h>
#include <unordered_map>
#include <fstream>
#include "../include/netparam.hpp"

using std::unordered_map;
using std::shared_ptr;

namespace lego_net {


class Net {

public:
    Net(){}

    /*! \brief forward and backward */
    void trainNet(shared_ptr<Blob>& X, 
                  shared_ptr<Blob>& Y,
                  NetParam& param,
                  std::string mode = "fb");

    /*! \brief test if all layers are right, be careful set reg to 0 */
    void testNet(NetParam& param);

    /*!
     * \brief set input data and ground truth
     * \param[in]  NetParam& param                   net parameters
     * \param[in]  vector<shared_ptr<Blob>>& X       X[0]: train data,  X[1]: val data
     * \param[in]  vector<shared_ptr<Blob>>& Y       Y[0]: train label, Y[1]: val label
     */
    void setup(NetParam& param,
                 vector<shared_ptr<Blob>>& X,
                 vector<shared_ptr<Blob>>& Y);

    /*! \brief train the net */
    void slove(NetParam& param);

    //void sampleInitData();

    /*! test num_grads of lnum th layer */
    void testLayer(NetParam& param, int lnum);

private:

    void _test_fc_layer(vector<shared_ptr<Blob>>& in,
                        vector<shared_ptr<Blob>>& grads,
                        shared_ptr<Blob>& dout); 

    void _test_conv_layer(vector<shared_ptr<Blob>>& in,
                         vector<shared_ptr<Blob>>& grads,
                         shared_ptr<Blob>& dout,
                         Param& param); 

    void _test_pool_layer(vector<shared_ptr<Blob>>& in,
                         vector<shared_ptr<Blob>>& grads,
                         shared_ptr<Blob>& dout,
                         Param& param); 

    void _test_relu_layer(vector<shared_ptr<Blob>>& in,
                         vector<shared_ptr<Blob>>& grads,
                         shared_ptr<Blob>& dout); 

    void _test_dropout_layer(vector<shared_ptr<Blob>>& in,
                         vector<shared_ptr<Blob>>& grads,
                         shared_ptr<Blob>& dout,
                         Param& param); 

    void _test_svm_layer(vector<shared_ptr<Blob>>& in,
                         shared_ptr<Blob>& dout); 

    void _test_softmax_layer(vector<shared_ptr<Blob>>& in,
                         shared_ptr<Blob>& dout); 

    /*! \brief save data names */
    vector<std::string> layers_;
    /*! \brief save data types */
    vector<std::string> ltype_;
    /*! \brief temporary loss score */
    double loss_;
    // train data 
    shared_ptr<Blob> X_train_;
    shared_ptr<Blob> Y_train_;
    // val data 
    shared_ptr<Blob> X_val_;
    shared_ptr<Blob> Y_val_;
    
    unordered_map<std::string, vector<shared_ptr<Blob>>> data_;
    unordered_map<std::string, vector<shared_ptr<Blob>>> grads_;
    unordered_map<std::string, vector<shared_ptr<Blob>>> num_grads_;

    /*! train or test */
    std::string type_;

    /*! train result */
    vector<double> loss_history_;
    vector<double> train_acc_history_;
    vector<double> val_acc_history_;

    /*! step cache */
    unordered_map<std::string, vector<shared_ptr<Blob>>> step_cache_;

    /*! \brief best model */
    unordered_map<std::string, vector<shared_ptr<Blob>>> best_model_;

}; // class Net

} // 

#endif
