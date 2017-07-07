# LegoNet
CNN框架设计（from zero to one）


目前主要完成CNN的基本部分。


目前主要目标为模块化各个层。

依赖：

'sudo apt-get install libopenblas-dev liblapack-dev'

armadillo:

'''
tar -Jxvf armadillo-7.200.2.tar.xz
cd armadillo-7.200.2
cmake .
make
sudo make install
'''

mnist:

' ./build/mnist  example/t10k-images-idx3-ubyte  example/t10k-labels-idx1-ubyte  example/mnist.json'
