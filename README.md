# LegoNet
CNN框架设计（from zero to one）


目前主要完成CNN的基本部分。

抽象：

![image](https://github.com/unsky/LegoNet/blob/master/framework.png)

目前主要目标为模块化各个层。
```
update 2017/7/19:通过设置calculate mode选择cpu模式或者gpu模式
upate 2017/8/6: blob模块尝试使用openblas
```

依赖：

```
sudo apt-get install libopenblas-dev liblapack-dev
```

armadillo(线性代数库):

```
wget http://sourceforge.net/projects/arma/files/armadillo-7.200.2.tar.xz

```

最新的版本可以在 http://arma.sourceforge.net/download.html 找到。

```
tar -Jxvf armadillo-7.200.2.tar.xz
cd armadillo-7.200.2
cmake .
make
sudo make install
```

安装 

```
安装legonet

make all

安装 debugall//debug 自定义层

make debugall

安装 测试mnist//mnist 例子

make mnist

```

测试mnist:
```
 ./data/mnist/get_mnist.sh

 ./build/mnist  data/mnist/t10k-images-idx3-ubyte  data/mnist/t10k-labels-idx1-ubyte  example/mnist.json
```
测试自定义层步骤
```
1. 在include/layers.hpp 中添加层定义
2. 在src/ 中定义自己的层
3. 在debug/debug_layers.hpp中添加要测试的层
4. 在debug/debug_layers/中添加要测试方法
5. 使用debug/debug进行测试
6. 在Makefile中的 debugall中添加自定义层的cpp
```

