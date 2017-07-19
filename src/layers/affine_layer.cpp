#include "../include/layer.hpp"

namespace lego_net {
    /*!
* \brief forward
*             X:        [N, C, Hx, Wx]
*             weight:   [F, C, Hw, Ww]
*             bias:     [F, 1, 1, 1]
*             out:      [N, F, 1, 1]
* \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
* \param[in]  const Param* param            params
* \param[out] Blob& out                     Y
*/


    /*!
    * \brief forward
    * Blob bottom[0]:                                                                 in[1]:weight
    *     _______          _______         __  _______________________         __      __  _______________________          __
    *  C /______/|   N    /______/|        |  |_______________________| __      |      |  |_______________________| __       |
    *   |------||| ······|------|||   ===> |          ...                |      |   *  |            ...              |       | . T() + b
    * H |------|||       |------|||   ===> |   _______________________    > N   |      |   _______________________    > F    |
    *   |------|/        |------|/         |_ |_______________________| _|     _|      |_ |_______________________| _|      _|
    *      W                                                                                         
    *   \___________  __________/             \___________  __________/                  \___________  __________/          
    *               \/                                    \/                                         \/
    *           [N,C,H,W]                               C*H*W                                       C*H*N
    *
    *             X:        [N, C, Hx, Wx]
    *             weight:   [F, C, Hw, Ww]
    *             bias:     [F, 1, 1, 1]
    *             out:      [N, F, 1, 1]
    * \param[in]  const vector<Blob*>& in       in[0]:X, in[1]:weights, in[2]:bias
    * \param[out] Blob& out                     Y
    */
void AffineLayer::cpu_forward(const vector<shared_ptr<Blob>>& in, shared_ptr<Blob>& out) {
    if (out) {
        out.reset();
    }
    int N = in[0]->get_N();
    int F = in[1]->get_N();

    mat x = in[0]->reshape();
    mat w = in[1]->reshape();
    mat b = in[2]->reshape();
    b = repmat(b, 1, N).t();
    mat ans = x * w.t() + b;
    mat2Blob(ans, out, F, 1, 1);

    return;
}

/*!
* \brief backward
*             in:       [N, C, Hx, Wx]
*             weight:   [F, C, Hw, Ww]
*             bias:     [F, 1, 1, 1]
*             dout:     [N, F, 1, 1]
* \param[in]  const Blob* dout              dout
* \param[in]  const vector<Blob*>& cache    cache[0]:X, cache[1]:weights, cache[2]:bias
* \param[out] vector<Blob*>& grads          grads[0]:dX, grads[1]:dW, grads[2]:db
*/
void AffineLayer::cpu_backward(shared_ptr<Blob>& dout,
                           const vector<shared_ptr<Blob>>& cache,
                           vector<shared_ptr<Blob>>& grads) {
    shared_ptr<Blob> dX;
    shared_ptr<Blob> dW;
    shared_ptr<Blob> db;

    int n = dout->get_N();

    shared_ptr<Blob> pX = cache[0];
    shared_ptr<Blob> pW = cache[1];
    shared_ptr<Blob> pb = cache[2];

    // calc grads
    // dX
    mat mat_dx = dout->reshape() * pW->reshape();
    mat2Blob(mat_dx, dX, pX->size());
    grads[0] = dX;
    // dW
    mat mat_dw = dout->reshape().t() * pX->reshape();
    mat2Blob(mat_dw, dW, (*pW).size());
    grads[1] = dW;
    // db
    mat mat_db = dout->reshape().t() * mat(n, 1, fill::ones);
    mat2Blob(mat_db, db, (*pb).size());
    grads[2] = db;

    return;
}
}