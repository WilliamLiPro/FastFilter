/***************************************************************
	>名称：关于不同滤波核的KII滤波算法
	>作者：李维鹏
	>联系方式：williamli_pro@163.com
	>采用不同滤波核执行KII滤波
	>要点：
	>1.高斯核KII滤波
	>2.sobel核KII滤波
	>3.DOG核KII滤波
	>4.标准人脸模板KII滤波

****************************************************************/

#pragma once

// C++标准库
#include <iostream>
#include <fstream>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

using namespace std;

/*	高斯核KII滤波函数
	输入：
	cv::Mat& image_in		输入图像
	cv::Mat& filte_kernel	滤波核
	const string data_type	输出图像数据类型（uchar char ushort short int float,默认值与image_in类型相同）
	输出：
	cv::Mat& image_out		输出图像
	*/
void theKIIgaussFilter(cv::Mat& image_in,cv::Mat& filte_kernel,cv::Mat& image_out,const string data_type);

/*	sobel核KII滤波函数
	输入：
	cv::Mat& image_in		输入图像
	cv::Mat& filte_kernel	滤波核
	const string data_type	输出图像数据类型（uchar char ushort short int float,默认值与image_in类型相同）
	输出：
	cv::Mat& image_out		输出图像
	*/
void theKIIsobelFilter(cv::Mat& image_in,cv::Mat& filte_kernel,cv::Mat& image_out,const string data_type);

/*	DOG核KII滤波函数
	输入：
	cv::Mat& image_in		输入图像
	cv::Mat& filte_kernel	滤波核
	const string data_type	输出图像数据类型（uchar char ushort short int float,默认值与image_in类型相同）
	输出：
	cv::Mat& image_out		输出图像
	*/
void theKIIdogFilter(cv::Mat& image_in,cv::Mat& filte_kernel,cv::Mat& image_out,const string data_type);

/*	标准人脸模板KII滤波函数
	输入：
	cv::Mat& image_in		输入图像
	cv::Mat& face_mask		标准人脸模板
	const string data_type	输出图像数据类型（uchar char ushort short int float,默认值与image_in类型相同）
	输出：
	cv::Mat& image_out		输出图像
	*/
void theKIIdogFilter(cv::Mat& image_in,cv::Mat& face_mask,cv::Mat& image_out,const string data_type);