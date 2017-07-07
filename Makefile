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

.PHONY : all
all : ofsmall mnist
	-mkdir build
	mv *.o *.a mnist ofsmall build

ofsmall : ofsmall.o libmnet.a
	$(CC) -o ofsmall ofsmall.o libmnet.a $(CXXFLAG)

ofsmall.o : lego_net.hpp ofsmall.cpp
	$(CC) -c  src/ofsmall.cpp $(CXXFLAG)

mnist : mnist.o libmnet.a
	$(CC) -o mnist mnist.o libmnet.a $(CXXFLAG)

mnist.o : lego_net.hpp mnist.cpp
	$(CC) -c example/mnist.cpp $(CXXFLAG)

libmnet.a : blob.o  net.o  netparam.o affine_layer.o conv_layer.o dropout_layer.o pool_layer.o relu_layer.o softmaxloss_layer.o svmloss_layer.o
	ar r libmnet.a blob.o  net.o netparam.o affine_layer.o conv_layer.o dropout_layer.o pool_layer.o relu_layer.o softmaxloss_layer.o svmloss_layer.o

blob.o layer.o net.o netparam.o: blob.hpp layer.hpp net.hpp  netparam.hpp test.hpp blob.cpp net.cpp netparam.cpp dropout_layer.cpp pool_layer.cpp relu_layer.cpp  softmaxloss_layer.cpp svmloss_layer.cpp affine_layer.cpp conv_layer.cpp
	$(CC) -c src/blob.cpp src/net.cpp src/netparam.cpp src/layers/dropout_layer.cpp src/layers/pool_layer.cpp src/layers/relu_layer.cpp  src/layers/relu_layer.cpp  src/layers/softmaxloss_layer.cpp src/layers/svmloss_layer.cpp src/layers/affine_layer.cpp src/layers/conv_layer.cpp $(CXXFLAG)

.PHONY: clean
clean:
	-rm -r build/
	-rm *.o *.a mnist ofsmall 
