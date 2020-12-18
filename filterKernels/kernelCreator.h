/***************************************************************
	>名称：滤波核生成函数集
	>作者：李维鹏
	>联系方式：williamli_pro@163.com
	>生成实验用，不同尺寸的各类滤波核（包括人脸模板）
	>要点：
	>1.生成高斯核
	>2.生成sobel核
	>3.生成DOG核
	>4.生成标准人脸模板

****************************************************************/

#pragma once

// C++标准库
#include <iostream>
#include <fstream>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

using namespace std;

/*	高斯核生成函数
	输入：
	int nx					滤波核x尺寸
	int ny					滤波核y尺寸
	float std_x			滤波核x标准差
	float std_y			滤波核y标准差
	输出：
	cv::Mat& out_kernel		高斯滤波核
	*/
void gaussKernelCreator(cv::Mat& out_kernel,int nx,int ny,float std_x,float std_y);

/*	sobel核生成函数(默认差分方向为y)
	输入：
	int nk					滤波核尺寸
	float std_x			高斯滤波核x标准差
	float dy				sobel核y差分
	输出：
	cv::Mat& out_kernel		高斯滤波核
	*/
void sobelKernelCreator(cv::Mat& out_kernel,int nk,float std_x,float dy);

/*	DOG核生成函数
	输入：
	int nk					滤波核尺寸
	float std_1			滤波核x标准差
	float std_2			滤波核y标准差
	输出：
	cv::Mat& out_kernel		高斯滤波核
	*/
void theDOGKernelCreator(cv::Mat& out_kernel,int nk,float std_1,float std_2);

/*	生成标准人脸模板
	输入：
	cv::Mat& pic_in			输入图像
	int width				输出宽度
	int height				输出高度
	输出：
	cv::Mat& out_mask		输出模板
	*/
void standardFaceCreator(cv::Mat& pic_in,cv::Mat& out_mask,int width,int height);