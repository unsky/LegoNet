#include "../include/layer.hpp"

namespace lego_net {



/*!
* \brief forward
*             X:        [N, C, 1, 1], usually the output of affine(fc) layer
*             Y:        [N, C, 1, 1], ground truth, with 1(true) or 0(false)
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:Y
* \param[out] double& loss                  loss
* \param[out] Blob** out                    out: dX
* \param[in]  int mode                      1: only forward, 0:forward and backward
*/
void SVMLossLayer::go(const vector<shared_ptr<Blob>>& in,
                      double& loss,
                      shared_ptr<Blob>& dout,
                      int mode) {
    if (dout) {
        dout.reset();
    }
    /*! let delta equals to 1 */
    double delta = 0.2;
    int N = in[0]->get_N();
    int C = in[0]->get_C();
    mat mat_x = in[0]->reshape();
    mat mat_y = in[1]->reshape();
    //mat_x.print("X:\n");
    //mat_y.print("Y:\n");

    /*! forward */
    mat good_x = repmat(arma::sum(mat_x % mat_y, 1), 1, C);
    mat mat_loss = (mat_x - good_x + delta);
    mat_loss.transform([](double e) {return e > 0 ? e : 0;});
    mat_y.transform([](double e) {return e ? 0 : 1;});
    mat_loss %= mat_y;
    loss = accu(mat_loss) / N;
    if (mode == 1)
        return;

    /*! backward */
    mat dx(mat_loss);
    dx.transform([](double e) {return e ? 1 : 0;});
    mat_y.transform([](double e) {return e ? 0 : 1;});
    mat sum_x = repmat(arma::sum(dx, 1), 1, C) % mat_y;
    dx = (dx - sum_x) / N;
    mat2Blob(dx, dout, in[0]->size());
    return;
}

} //namespace lego_net
