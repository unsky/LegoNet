#include "../include/layer.hpp"

namespace lego_net {
    /*!
* \brief convolutional layer forward
*             X:        [N, C, Hx, Wx]
*             weight:   [F, C, Hw, Ww]
*             bias:     [F, 1, 1, 1]
*             out:      [N, F, (Hx+pad*2-Hw)/stride+1, (Wx+pad*2-Ww)/stride+1]
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
* \param[in]  const ConvParam* param        conv params: stride, pad
* \param[out] Blob** out                    Y
*/


    /*!
    * \brief forward
    *  Blob bottom[0]:
    *     _______          _______                                               _______          _______
    *  C /______/|   N    /______/|                                           C /______/|   N*F  /______/| 
    *   |------||| ······|------|||                                            |------||| ······|------|||
    * H |------|||       |------|||    *   F个kernel size为(n,n)的卷积核  =  Hw |------|||       |------||| Hw
    *   |------|/        |------|/                                             |------|/        |------|/
    *      W                                                                      Ww               Ww
    *   \___________  __________/
    *               \/
    *            [N,C,H,W]  
    *
    *             X:        [N, C, Hx, Wx]
    *             weight:   [F, C, Hw, Ww]
    *             bias:     [F, 1, 1, 1]
    *             out:      [N, F, (Hx+pad*2-Hw)/stride+1, (Wx+pad*2-Ww)/stride+1]
    * \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
    * \param[in]  const ConvParam* param        conv params
    * \param[out] Blob& out                     Y
    */
void ConvLayer::cpu_forward(const vector<shared_ptr<Blob>>& in,
                        shared_ptr<Blob>& out,
                        Param& param) {
    if (out) {
        out.reset();
    }
    assert(in[0]->get_C() == in[1]->get_C());
    int N = in[0]->get_N();
    int F = in[1]->get_N();
    int C = in[0]->get_C();
    int Hx = in[0]->get_H();
    int Wx = in[0]->get_W();
    int Hw = in[1]->get_H();
    int Ww = in[1]->get_W();

    // calc Hy, Wy
    int Hy = (Hx + param.conv_pad*2 -Hw) / param.conv_stride + 1;
    int Wy = (Wx + param.conv_pad*2 -Ww) / param.conv_stride + 1;

    out.reset(new Blob(N, F, Hy, Wy));
    Blob padX = (*in[0]).pad(param.conv_pad);

    for (int n = 0; n < N; ++n) {
        for (int f = 0; f < F; ++f) {
            for (int hh = 0; hh < Hy; ++hh) {
                for (int ww = 0; ww < Wy; ++ww) {
                    cube window = padX[n](span(hh * param.conv_stride, hh * param.conv_stride + Hw - 1),
                                            span(ww * param.conv_stride, ww * param.conv_stride + Ww - 1),
                                            span::all);
                    (*out)[n](hh, ww, f) = accu(window % (*in[1])[f]) + as_scalar((*in[2])[f]);
                }
            }
        }
    }
    return;
}

/*!
* \brief backward
*             in:       [N, C, Hx, Wx]
*             weight:   [F, C, Hw, Ww]
*             bias:     [F, 1, 1, 1]
*             out:      [N, F, (Hx+pad*2-Hw)/stride+1, (Wx+pad*2-Ww)/stride+1]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X, cache[1]:weights, cache[2]:bias
* \param[out] vector<Blob*>& grads          grads[0]:dX, grads[1]:dW, grads[2]:db
*/



void ConvLayer::cpu_backward(shared_ptr<Blob>& dout,
                         const vector<shared_ptr<Blob>>& cache,
                         vector<shared_ptr<Blob>>& grads,
                         Param& param) {
    int N = cache[0]->get_N();
    int F = cache[1]->get_N();
    int C = cache[0]->get_C();
    int Hx = cache[0]->get_H();
    int Wx = cache[0]->get_W();
    int Hw = cache[1]->get_H();
    int Ww = cache[1]->get_W();
    int Hy = dout->get_H();
    int Wy = dout->get_W();
    assert(C == cache[1]->get_C());
    assert(F == cache[2]->get_N());

    shared_ptr<Blob> dX(new Blob(cache[0]->size(), TZEROS));
    shared_ptr<Blob> dW(new Blob(cache[1]->size(), TZEROS));
    shared_ptr<Blob> db(new Blob(cache[2]->size(), TZEROS));

    Blob pad_dX(N, C, Hx + param.conv_pad*2, Wx + param.conv_pad*2, TZEROS);
    Blob pad_X = (*cache[0]).pad(1);

    for (int n = 0; n < N; ++n) {
        for (int f = 0; f < F; ++f) {
            for (int hh = 0; hh < Hy; ++hh) {
                for (int ww = 0; ww < Wy; ++ww) {
                    cube window = pad_X[n](span(hh * param.conv_stride,  hh * param.conv_stride + Hw - 1),
                                            span(ww * param.conv_stride, ww * param.conv_stride + Ww - 1),
                                            span::all);
                    (*db)[f](0, 0, 0) += (*dout)[n](hh, ww, f);
                    (*dW)[f] += window * (*dout)[n](hh, ww, f);
                    pad_dX[n](span(hh * param.conv_stride, hh * param.conv_stride + Hw - 1),
                        span(ww * param.conv_stride, ww * param.conv_stride + Ww - 1),
                        span::all) += (*cache[1])[f] * (*dout)[n](hh, ww, f);
                }
            }
        }
    }
    *dX = pad_dX.dePad(param.conv_pad);
    grads[0] = dX;
    grads[1] = dW;
    grads[2] = db;

    return;
}

}