#include "../include/layer.hpp"

namespace lego_net {

/*!
* \brief forward
*             X:        [N, C, Hx, Wx]
*             out:      [N, C, Hx/2, Wx/2]
* \param[in]  const vector<Blob*>& in       in[0]:X
* \param[in]  const Param* param        conv params
* \param[out] Blob& out                     Y
*/
void PoolLayer::cpu_forward(const vector<shared_ptr<Blob>>& in,
                        shared_ptr<Blob>& out,
                        Param& param) {
    if (out) {
        out.reset();
    }
    int N = (*in[0]).get_N();
    int C = (*in[0]).get_C();
    int Hx = (*in[0]).get_H();
    int Wx = (*in[0]).get_W();
    int height = param.pool_height;
    int width = param.pool_width;
    int stride = param.pool_stride;

    int Hy = (Hx - height) / stride + 1;
    int Wy = (Wx - width) / stride + 1;

    out.reset(new Blob(N, C, Hy, Wy));

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int hh = 0; hh < Hy; ++hh) {
                for (int ww = 0; ww < Wy; ++ww) {
                    (*out)[n](hh, ww, c) = (*in[0])[n](span(hh * stride, hh * stride + height - 1),
                                                        span(ww * stride, ww * stride + width - 1),
                                                        span(c, c)).max();
                }
            }
        }
    }
    return;
}

/*!
* \brief backward
*             cache:    [N, C, Hx, Wx]
*             dout:     [N, F, Hx/2, Wx/2]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X
* \param[out] vector<Blob*>& grads          grads[0]:dX
*/
void PoolLayer::cpu_backward(shared_ptr<Blob>& dout,
                         const vector<shared_ptr<Blob>>& cache,
                         vector<shared_ptr<Blob>>& grads,
                         Param& param) {
    int N = cache[0]->get_N();
    int C = cache[0]->get_C();
    int Hx = cache[0]->get_H();
    int Wx = cache[0]->get_W();
    int Hy = dout->get_H();
    int Wy = dout->get_W();
    int height = param.pool_height;
    int width = param.pool_width;
    int stride = param.pool_stride;

    shared_ptr<Blob> dX(new Blob(cache[0]->size(), TZEROS));

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int hh = 0; hh < Hy; ++hh) {
                for (int ww = 0; ww < Wy; ++ww) {
                    mat window = (*cache[0])[n](span(hh * stride, hh * stride + height - 1),
                                    span(ww * stride, ww * stride + width - 1),
                                    span(c, c));
                    double maxv = window.max();
                    mat mask = conv_to<mat>::from(maxv == window);
                    (*dX)[n](span(hh * stride, hh * stride + height - 1),
                            span(ww * stride, ww * stride + width - 1),
                            span(c, c)) += mask * (*dout)[n](hh, ww, c);
                }
            }
        }
    }
    grads[0] = dX;
    return;
}
}