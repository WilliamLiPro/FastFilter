/***************************************************************
	>���ƣ��˲������ɺ�����
	>���ߣ���ά��
	>��ϵ��ʽ��williamli_pro@163.com
	>����ʵ���ã���ͬ�ߴ�ĸ����˲��ˣ���������ģ�壩
	>Ҫ�㣺
	>1.���ɸ�˹��
	>2.����sobel��
	>3.����DOG��
	>4.���ɱ�׼����ģ��

****************************************************************/

#pragma once

// C++��׼��
#include <iostream>
#include <fstream>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

using namespace std;

/*	��˹�����ɺ���
	���룺
	int nx					�˲���x�ߴ�
	int ny					�˲���y�ߴ�
	float std_x			�˲���x��׼��
	float std_y			�˲���y��׼��
	�����
	cv::Mat& out_kernel		��˹�˲���
	*/
void gaussKernelCreator(cv::Mat& out_kernel,int nx,int ny,float std_x,float std_y);

/*	sobel�����ɺ���(Ĭ�ϲ�ַ���Ϊy)
	���룺
	int nk					�˲��˳ߴ�
	float std_x			��˹�˲���x��׼��
	float dy				sobel��y���
	�����
	cv::Mat& out_kernel		��˹�˲���
	*/
void sobelKernelCreator(cv::Mat& out_kernel,int nk,float std_x,float dy);

/*	DOG�����ɺ���
	���룺
	int nk					�˲��˳ߴ�
	float std_1			�˲���x��׼��
	float std_2			�˲���y��׼��
	�����
	cv::Mat& out_kernel		��˹�˲���
	*/
void theDOGKernelCreator(cv::Mat& out_kernel,int nk,float std_1,float std_2);

/*	���ɱ�׼����ģ��
	���룺
	cv::Mat& pic_in			����ͼ��
	int width				������
	int height				����߶�
	�����
	cv::Mat& out_mask		���ģ��
	*/
void standardFaceCreator(cv::Mat& pic_in,cv::Mat& out_mask,int width,int height);