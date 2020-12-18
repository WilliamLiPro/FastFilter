/***************************************************************
	>名称：图像的积分运算
	>作者：李维鹏
	>联系方式：williamli_pro@163.com
	>实现对图像的快速完整积分
	>技术要点：
	>1.uchar型多通道图像的快速完整积分
	>2.一般多通道图像的快速完整积分

****************************************************************/

#pragma once

// C++标准库
#include <iostream>
#include <fstream>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

using namespace std;

//	************************* 编写函数的头文件 ***************************


/*	uchar(char)型多通道图像的快速完整积分
	输入：
	cv::Mat& image_in		输入图像（uchar(char)型）
	int data_type			数据类型（0 uchar,1 char）
	输出：
	cv::Mat& image_out		输出图像（int型）
	当函数返回false时，程序出错*/
bool fastUcharImageIntegration(cv::Mat& image_in,int data_type,cv::Mat& image_out);

/*	一般多通道图像的快速完整积分	
	输入：
	cv::Mat& image_in		输入图像（任意类型）
	输出：
	cv::Mat& image_out		输出图像（float型）
	当函数返回false时，程序出错*/
bool fastNormalImageIntegration(cv::Mat& image_in,cv::Mat& image_out);

/*	图像积分运算测试函数	*/
bool imageIntegrationTest();
