#ifndef LEGO_NET_SLOVER_HPP
#define LEGO_NET_SLOVER_HPP

#include "blob.hpp"
#include "net.hpp"
#include "netparam.hpp"
namespace lego_net{
class Slover{

public:
    Slover(){}
    void slove(NetParam& param, vector<shared_ptr<Blob>> X, vector<shared_ptr<Blob>> Y);

private:

    // train data 
    shared_ptr<Blob> X_train_;
    shared_ptr<Blob> Y_train_;
    // val data 
    shared_ptr<Blob> X_val_;
    shared_ptr<Blob> Y_val_;




};//end class

}//end lego_net
#endif // !LEGO_NET_SLOVER_HPP