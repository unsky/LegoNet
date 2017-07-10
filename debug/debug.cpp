
#include <iostream>
#include <armadillo>
#include <../include/lego_net.hpp>
#include <../debug/debug_layers.hpp>
using namespace debug;
int main()
{
    //testSharedptr();
    //testArma();
    //testBlob();
   // testAffineLayer();
    //testConvLayer();
    //testPoolLayer();
    //testRelu();
    //testDropout();
    //testSoftmax();
    //testSVM();
    //testTest();
    //int n = 3;
    //testNet();
    //while (n--) {
    //    mat a(5,5,fill::randn);
    //    a.print("a\n");
    //}
    //Blob b(4,2,2,2,TRANDN);
    //b.reshape().print();
    testSampleNetTrain();

    return 0;
}