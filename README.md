## DEEPSORT

#### 说明 

[原项目地址](https://github.com/weixu000/libtorch-yolov3-deepsort)
 
此项目以学习Deepsort原理为主，在原项目的基础上进行小修改，编译出可供C++与python两个语言进行调用的库，方便工程的小伙伴直接使用<br>

感谢原作者提供的代码！<br>

`demo` 文件夹下提供了C++与Python两个版本的调用示例，C++使用的是pytorch的yolo3, python使用的是mxnet 
`tracking` 文件夹下为Deepsort源码

#### 依赖

Ubuntu下以测试完成

1. opencv3 以及更高版本 
2. libtorch [下载地址 V1.51](https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu101.zip) 下载完成后复制进`thirdpart`文件夹即可
3. pybind11 已被添加进`thirdpart`文件夹 无需安装

模型权重文件：
- [YOLOv3](https://pjreddie.com/media/files/yolov3.weights)

#### 编译
1. 编译c++ demo以及动态库
```
mkdir build
cd build
cmake -DBUILD_PYTHON_PACKAGE=OFF ..
make -j2
```

运行c++ demo：
```
cd build/bin
./demo xxx.mp4
```

**注意**：
运行demo请先修改`demo/c++/main.cpp`中的yolo权重路径

2. 编译python库
```
mkdir build
cd build
cmake -DBUILD_PYTHON_PACKAGE=ON ..
make -j2
```

运行python demo
```
cd demo/python
python test.py
```

**注意**：
1. 运行`demo/python/test.py`需要将编译好的`tracking.cpython-3xm-x86_64-linux-gnu.so`复制到`demo/python`目录下，`tracking.cpython-3xm-x86_64-linux-gnu.so`一般位于`build/tracking`下, 并且注意`.so`的python版本与python版本对应，否则可能出现`import error`
2. 修改test.py中对应的参数 例如`video_path` `width` `height`， `width` `heigth`是ssd预处理图片缩放后的宽高大小
3. 需要安装opencv-python mxnet-cuxx(对应你的cuda版本 例如cu10.1--mxnet-cu101) numpy gluoncv
```
pip install mxnet-cuxx opencv-python numpy gluoncv
```


