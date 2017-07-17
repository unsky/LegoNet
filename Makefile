# LegoNet makefile

CC := g++
JSON_INC = $(shell pkg-config jsoncpp --cflags)
JSON_LIB = $(shell pkg-config jsoncpp --libs)
ARMADILLO_INC = -I/usr/include
ARMADILLO_LIB = -larmadillo
CXXFLAG = -I./include -std=c++11 -g
CXXFLAG += $(JSON_INC)
CXXFLAG += $(JSON_LIB)
CXXFLAG += $(ARMADILLO_INC)
CXXFLAG += $(ARMADILLO_LIB)

vpath %.hpp include
vpath %.cpp src
vpath %.cpp src/layers
vpath %.cpp example
vpath %.cpp debug
vpath %.cpp debug/debug_layers
vpath %.hpp debug

.PHONY : all
all : debugall.o  debugall mnist
	-mkdir build
	-mkdir build/legonet
	mv  mnist debugall build
	mv *.o *.a build/legonet
debugall : debugall.o libleogonet.a
	$(CC) -o debugall debug.o AffineLayer_debug.o ConvLayer_debug.o  Dropout_debug.o PoolLayer_debug.o Relu_debug.o SampleNetTrain_debug.o  Softmax_debug.o SVM_debug.o libleogonet.a $(CXXFLAG)

debugall.o : lego_net.hpp debug_layers.hpp debug.cpp 
	$(CC) -c  debug/debug.cpp debug/debug_layers/AffineLayer_debug.cpp debug/debug_layers/ConvLayer_debug.cpp  debug/debug_layers/Dropout_debug.cpp debug/debug_layers/PoolLayer_debug.cpp debug/debug_layers/Relu_debug.cpp debug/debug_layers/SampleNetTrain_debug.cpp debug/debug_layers/Softmax_debug.cpp  debug/debug_layers/SVM_debug.cpp $(CXXFLAG)

mnist : mnist.o libleogonet.a
	$(CC) -o mnist mnist.o libleogonet.a $(CXXFLAG)

mnist.o : lego_net.hpp mnist.cpp
	$(CC) -c example/mnist.cpp $(CXXFLAG)

libleogonet.a : blob.o  net.o  slover.o netparam.o affine_layer.o conv_layer.o dropout_layer.o pool_layer.o relu_layer.o softmaxloss_layer.o svmloss_layer.o
	ar r libleogonet.a blob.o slover.o net.o netparam.o affine_layer.o conv_layer.o dropout_layer.o pool_layer.o relu_layer.o softmaxloss_layer.o svmloss_layer.o

blob.o layer.o net.o netparam.o: blob.hpp layer.hpp net.hpp slover.hpp netparam.hpp test.hpp blob.cpp net.cpp netparam.cpp dropout_layer.cpp pool_layer.cpp relu_layer.cpp  softmaxloss_layer.cpp svmloss_layer.cpp affine_layer.cpp conv_layer.cpp
	$(CC) -c src/blob.cpp src/net.cpp src/slover.cpp src/netparam.cpp src/layers/dropout_layer.cpp src/layers/pool_layer.cpp src/layers/relu_layer.cpp  src/layers/relu_layer.cpp  src/layers/softmaxloss_layer.cpp src/layers/svmloss_layer.cpp src/layers/affine_layer.cpp src/layers/conv_layer.cpp $(CXXFLAG)
	


.PHONY: clean
clean:
	-rm -r build/
	-rm *.o *.a mnist debugall
