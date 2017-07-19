#include "../include/layer.hpp"

namespace lego_net {



/*!
* \brief forward
*             X:        [N, C, Hx, Wx]
*             out:      [N, C, Hx, Wx]
* \param[in]  const vector<Blob*>& in       in[0]:X
* \param[in]  Param& param                  int mode, double p, int seed, Blob *mask
* \param[out] Blob& out                     Y
*/
void DropoutLayer::cpu_forward(const vector<shared_ptr<Blob>>& in,
                           shared_ptr<Blob>& out,
                           Param& param) {
    if (out) {
        out.reset();
    }
    int mode = param.drop_mode;
    double p = param.drop_p;
    assert(0 <= p && p <= 1);
    assert(0 <= mode && mode <= 3);
    int seed;
    /*! train mode */
    if ((mode & 1) == 1) {
        if ((mode & 2) == 2) {
            seed = param.drop_seed;
        }
        shared_ptr<Blob> mask(new Blob(seed, in[0]->size(), TRANDU));
        (*mask).smallerIn(p);
        Blob in_mask = (*in[0]) * (*mask);
        out.reset(new Blob(in_mask / p));
        if (param.drop_mask) {
            param.drop_mask.reset();
        }
        param.drop_mask = mask;
    }
    else {
        /*! test mode */
        out.reset(new Blob(*in[0]));
    }
    return;
}

/*!
* \brief backward
*             in:       [N, C, Hx, Wx]
*             dout:     [N, F, Hx, Wx]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X
* \param[in]  Param& param                  int mode, double p, int seed, Blob *mask
* \param[out] vector<Blob*>& grads          grads[0]:dX
*/
void DropoutLayer::cpu_backward(shared_ptr<Blob>& dout,
                            const vector<shared_ptr<Blob>>& cache,
                            vector<shared_ptr<Blob>>& grads,
                            Param& param) {
    shared_ptr<Blob> dX(new Blob((*dout)));
    int mode = param.drop_mode;
    assert(0 <= mode && mode <= 3);
    if ((mode & 1) == 1) {
        Blob dx_mask = (*dX) * (*param.drop_mask);
        *dX = dx_mask / param.drop_p;
    }
    grads[0] = dX;
    return;
}
}