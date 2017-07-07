#include "../include/layer.hpp"

namespace lego_net {



/*!
* \brief forward
*             X:        [N, C, 1, 1], usually the output of affine(fc) layer
*             Y:        [N, C, 1, 1], ground truth, with 1(true) or 0(false)
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:Y
* \param[out] double& loss                  loss
* \param[out] Blob** out                    out: dX
*/
void SoftmaxLossLayer::go(const vector<shared_ptr<Blob>>& in,
                          double& loss,
                          shared_ptr<Blob>& dout,
                          int mode) {
    //Blob X(*in[0]);
    //Blob Y(*in[1]);
    if (dout) {
        dout.reset();
    }
    int N = in[0]->get_N();
    int C = in[0]->get_C();
    int H = in[0]->get_H();
    int W = in[0]->get_W();
    assert(H == 1 && W == 1);

    mat mat_x = in[0]->reshape();
    mat mat_y = in[1]->reshape();

    /*! forward */
    mat row_max = repmat(arma::max(mat_x, 1), 1, C);
    mat_x = arma::exp(mat_x - row_max);
    mat row_sum = repmat(arma::sum(mat_x, 1), 1, C);
    mat e = mat_x / row_sum;
    //e.print("e:\n");
    //mat rrs = arma::sum(e, 1);
    //rrs.print("rrs:\n");
    mat prob = -arma::log(e);
    //prob.print("prob:\n");
    //(prob%mat_y).print("gg:\n");
    /*! loss should near -log(1/C) */
    loss = accu(prob % mat_y) / N;
    /*! only forward */
    if (mode == 1)
        return;

    /*! backward */
    mat dx = e - mat_y;
    dx /= N;
    mat2Blob(dx, dout, (*in[0]).size());
    return;
}
}