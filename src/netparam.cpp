#include "../include/net.hpp"
#include "../include/netparam.hpp"

namespace lego_net {
    void NetParam::readNetParam(std::string file) {
    std::ifstream ifs;
    ifs.open(file);
    assert(ifs.is_open());
    Json::Reader reader;
    Json::Value value;
    if (reader.parse(ifs, value)) {
        if (!value["train"].isNull()) {
            auto &tparam = value["train"];
            this->lr = tparam["learning rate"].asDouble();
            this->lr_decay = tparam["lr decay"].asDouble();
            this->update = tparam["update method"].asString();
            this->momentum = tparam["momentum parameter"].asDouble();
            this->num_epochs = tparam["num epochs"].asInt();
            this->use_batch = tparam["use batch"].asBool();
            this->batch_size = tparam["batch size"].asInt();
            this->reg = tparam["reg"].asDouble();
            this->acc_frequence = tparam["acc frequence"].asInt();
            this->acc_update_lr = tparam["frequence update"].asBool();
        }
        if (!value["net"].isNull()) {
            auto &nparam = value["net"];
            for (int i = 0; i < (int)nparam.size(); ++i) {
                auto &ii = nparam[i];
                this->layers.push_back(ii["name"].asString());
                this->ltypes.push_back(ii["type"].asString());
                if (ii["type"].asString() == "Conv") {
                    int num = ii["kernel num"].asInt();
                    int width = ii["kernel width"].asInt();
                    int height = ii["kernel height"].asInt();
                    int pad = ii["pad"].asInt();
                    int stride = ii["stride"].asInt();
                    this->params[ii["name"].asString()].setConvParam(stride, pad, width, height, num);
                }
                if (ii["type"].asString() == "Pool") {
                    int stride = ii["stride"].asInt();
                    int width = ii["kernel width"].asInt();
                    int height = ii["kernel height"].asInt();
                    this->params[ii["name"].asString()].setPoolParam(stride, width, height);
                }
                if (ii["type"].asString() == "Fc") {
                    int num = ii["kernel num"].asInt();
                    this->params[ii["name"].asString()].fc_kernels = num;
                }
            }
        }
    }
}
}
