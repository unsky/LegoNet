#ifndef DEBUG_LAYERS_HPP_
#define DEBUG_LAYERS_HPP_
#include <iostream>
#include <armadillo>
#include <../include/lego_net.hpp>

using namespace arma;
using namespace lego_net;
using std::shared_ptr;
namespace debug{
    //affine layer
    void testAffineLayer();
    void testConvLayer();
    void testPoolLayer();
    void testRelu();
    void testDropout();
    void testSoftmax();
    void testSVM();
    void testSampleNetTrain();


}// end debug namespace
#endif