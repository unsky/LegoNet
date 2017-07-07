
#ifndef   LEGO_NET_NETPARAM_HPP_
#define LEGO_NET_NETPARAM_HPP_
#include "blob.hpp"
#include "layer.hpp"
#include "test.hpp"
#include <json/json.h>
#include <unordered_map>
#include <fstream>
using std::unordered_map;
namespace lego_net {
struct NetParam {
    /*! methods of update net parameters, sgd/momentum/... */
    std::string update;
    /*! learning rate */
    double lr;
    double lr_decay;
    /*! momentum parameter */
    double momentum;
    int num_epochs;
    /*! whether use batch size */
    bool use_batch;
    int batch_size;
    /*! regulazation parameter */
    double reg;
    /*! \brief acc_frequence, how many iterations to check val_acc and train_acc */
    int acc_frequence;
    bool acc_update_lr;
    vector<std::string> layers;
    vector<std::string> ltypes;
    unordered_map<std::string, Param> params;

    void readNetParam(std::string file);
};
}
#endif