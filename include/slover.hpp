#ifndef LEGO_NET_SLOVER_HPP
#define LEGO_NET_SLOVER_HPP

#include "blob.hpp"
#include "net.hpp"
#include "netparam.hpp"
namespace lego_net{
class Slover{

public:
    Slover(NetParam& param){
    layers_ = param.layers;
    ltype_ = param.ltypes;
    for (int i = 0; i < (int)layers_.size(); ++i) {
        data_[layers_[i]] = vector<shared_ptr<Blob>>(3);
        grads_[layers_[i]] = vector<shared_ptr<Blob>>(3);
        step_cache_[layers_[i]] = vector<shared_ptr<Blob>>(3);
        best_model_[layers_[i]] = vector<shared_ptr<Blob>>(3);
    }
    }
    void slove(NetParam& param, vector<shared_ptr<Blob>> X, vector<shared_ptr<Blob>> Y);

    void trainNet(shared_ptr<Blob>& X, 
                  shared_ptr<Blob>& Y,
                  NetParam& param,
                  std::string mode = "fb");
private:

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
};//end class

}//end lego_net
#endif // !LEGO_NET_SLOVER_HPP