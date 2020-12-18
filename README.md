Fast Filter version 0.1
=======================================

Fast Filter based on kernel decomposition and integral image.  
This program runs on CPU

For more information, please contact author by: williamli_pro@163.com  

# System Requirements
Operating systems: Windows or Linux  
Dependencies: C++, OpenCV (version >= 2.4.1) 

# Demo
After installed all of the requirements, please run runTest.cpp

# License
Apache-2.0

# Next version
Next version will be accelerated by L1 cache optimization and AVX256 instruction set, which will achieve a predicable 20x acceleration.

有损快速滤波
=======================================

三年前的小工作  

1. 对滤波核进行常数+低秩分解，对展开后的每个一维滤波器进一步泰勒展开  
2. 使用积分图加速泰勒展开后的滤波过程  

如有问题请联系：williamli_pro@163.com  

# 系统要求
操作系统: Windows or Linux  
依赖项: C++, OpenCV (version >= 2.4.1) 

# 演示程序
安装所有依赖项后，运行 runTest.cpp

# 版本预告
下一版本将使用 L1 cache 优化和 AVX256指令集加速，预计将获得20x的加速效果

# License
Apache-2.0