#include "../include/layer.hpp"

namespace lego_net {

/*!
* \brief forward, out = max(0, X)
*             X:        [N, C, Hx, Wx]
*             out:      [N, C, Hx, Wx]
* \param[in]  const vector<Blob*>& in       in[0]:X
* \param[out] Blob& out                     Y
*/
void ReluLayer::forward(const vector<shared_ptr<Blob>>& in,
                        shared_ptr<Blob>& out) {
    if (out) {
        out.reset();
    }
    out.reset(new Blob(*in[0]));
    (*out).maxIn(0);
    return;
}

/*!
* \brief backward, dX = dout .* (X > 0)
*             in:       [N, C, Hx, Wx]
*             dout:     [N, F, Hx, Wx]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X
* \param[out] vector<Blob*>& grads          grads[0]:dX
*/
void ReluLayer::backward(shared_ptr<Blob>& dout,
                         const vector<shared_ptr<Blob>>& cache,
                         vector<shared_ptr<Blob>>& grads) {
    shared_ptr<Blob> dX(new Blob(*cache[0]));
    int N = cache[0]->get_N();
    for (int i = 0; i < N; ++i) {
        (*dX)[i].transform([](double e) {return e > 0 ? 1 : 0;});
    }
    (*dX) = (*dout) * (*dX);
    grads[0] = dX;
    return;
}
}