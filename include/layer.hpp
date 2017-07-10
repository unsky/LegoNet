
#ifndef LEGO_NET_LAYER_HPP_
#define LEGO_NET_LAYER_HPP_

#include "blob.hpp"
#include <memory>
using std::vector;
using std::shared_ptr;

namespace lego_net {

/*! layer parameters */
struct Param {
    Param() : conv_stride(0), conv_pad(0) {}
    /*! \brief conv param */
    int conv_stride;
    int conv_pad;
    int conv_width;
    int conv_height;
    int conv_kernels;
    inline void setConvParam(int stride, int pad, int width, int height, int kernels) {
        conv_stride = stride;
        conv_pad = pad;
        conv_width = width;
        conv_height = height;
        conv_kernels = kernels;
    }
    /*! \brief pool param */
    int pool_stride;
    int pool_width;
    int pool_height;
    inline void setPoolParam(int stride, int width, int height) {
        pool_stride = stride;
        pool_width = width;
        pool_height = height;
    }
    /*! \brief dropout param */
    /*! if the most right bit is 1 use train mode, else use test mode;
     *  if the second bit from right is 1, use random seed; else use selected seed. */
    int drop_mode;
    double drop_p;
    int drop_seed;
    shared_ptr<Blob> drop_mask;
    inline void setDropoutpParam(int mode, double pp, int s) {
        drop_mode = mode;
        drop_p = pp;
        drop_seed = s;
        drop_mask.reset();
    }
    /*! fc parameters */
    int fc_kernels;
};

// affine_layer
class AffineLayer {
public:
    AffineLayer() {}
    ~AffineLayer() {}
    static void forward(const vector<shared_ptr<Blob>>& in,
                        shared_ptr<Blob>& out);

    static void backward(shared_ptr<Blob>& dout,
                         const vector<shared_ptr<Blob>>& cache,
                         vector<shared_ptr<Blob>>& grads);
};

//conv_layer
class ConvLayer {
public:
    ConvLayer() {}
    ~ConvLayer() {}

    static void forward(const vector<shared_ptr<Blob>>& in,
                        shared_ptr<Blob>& out,
                        Param& param);

    static void backward(shared_ptr<Blob>& dout,
                         const vector<shared_ptr<Blob>>& cache,
                         vector<shared_ptr<Blob>>& grads,
                         Param& param);
};

//pool_layer
class PoolLayer {
public:
    PoolLayer() {}
    ~PoolLayer() {}

    static void forward(const vector<shared_ptr<Blob>>& in,
                        shared_ptr<Blob>& out,
                        Param& param);

    static void backward(shared_ptr<Blob>& dout,
                         const vector<shared_ptr<Blob>>& cache,
                         vector<shared_ptr<Blob>>& grads,
                         Param& param);
};

//relu_layer
class ReluLayer {
public:
    ReluLayer() {}
    ~ReluLayer() {}
    static void forward(const vector<shared_ptr<Blob>>& in,
                        shared_ptr<Blob>& out);
    static void backward(shared_ptr<Blob>& dout,
                         const vector<shared_ptr<Blob>>& cache,
                         vector<shared_ptr<Blob>>& grads);
};

//dropout_layer
class DropoutLayer {
public:
    DropoutLayer() {}
    ~DropoutLayer() {}


    static void forward(const vector<shared_ptr<Blob>>& in,
                        shared_ptr<Blob>& out,
                        Param& param);
    static void backward(shared_ptr<Blob>& dout,
                         const vector<shared_ptr<Blob>>& cache,
                         vector<shared_ptr<Blob>>& grads,
                         Param& param);
};

//softmaxloss_layer
class SoftmaxLossLayer {
public:
    SoftmaxLossLayer() {}
    ~SoftmaxLossLayer() {}
    static void go(const vector<shared_ptr<Blob>>& in,
                   double& loss,
                   shared_ptr<Blob>& dout,
                   int mode = 0);
};
//svmloss_layer
class SVMLossLayer {
public:
    SVMLossLayer() {}
    ~SVMLossLayer() {}
    static void go(const vector<shared_ptr<Blob>>& in,
                   double& loss,
                   shared_ptr<Blob>& dout, 
                   int mode = 0);
};

} // namespace lego_net

#endif // LEGO_NET_LAYER_