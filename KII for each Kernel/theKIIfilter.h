/***************************************************************
	>���ƣ����ڲ�ͬ�˲��˵�KII�˲��㷨
	>���ߣ���ά��
	>��ϵ��ʽ��williamli_pro@163.com
	>���ò�ͬ�˲���ִ��KII�˲�
	>Ҫ�㣺
	>1.��˹��KII�˲�
	>2.sobel��KII�˲�
	>3.DOG��KII�˲�
	>4.��׼����ģ��KII�˲�

****************************************************************/

#pragma once

// C++��׼��
#include <iostream>
#include <fstream>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

using namespace std;

/*	��˹��KII�˲�����
	���룺
	cv::Mat& image_in		����ͼ��
	cv::Mat& filte_kernel	�˲���
	const string data_type	���ͼ���������ͣ�uchar char ushort short int float,Ĭ��ֵ��image_in������ͬ��
	�����
	cv::Mat& image_out		���ͼ��
	*/
void theKIIgaussFilter(cv::Mat& image_in,cv::Mat& filte_kernel,cv::Mat& image_out,const string data_type);

/*	sobel��KII�˲�����
	���룺
	cv::Mat& image_in		����ͼ��
	cv::Mat& filte_kernel	�˲���
	const string data_type	���ͼ���������ͣ�uchar char ushort short int float,Ĭ��ֵ��image_in������ͬ��
	�����
	cv::Mat& image_out		���ͼ��
	*/
void theKIIsobelFilter(cv::Mat& image_in,cv::Mat& filte_kernel,cv::Mat& image_out,const string data_type);

/*	DOG��KII�˲�����
	���룺
	cv::Mat& image_in		����ͼ��
	cv::Mat& filte_kernel	�˲���
	const string data_type	���ͼ���������ͣ�uchar char ushort short int float,Ĭ��ֵ��image_in������ͬ��
	�����
	cv::Mat& image_out		���ͼ��
	*/
void theKIIdogFilter(cv::Mat& image_in,cv::Mat& filte_kernel,cv::Mat& image_out,const string data_type);

/*	��׼����ģ��KII�˲�����
	���룺
	cv::Mat& image_in		����ͼ��
	cv::Mat& face_mask		��׼����ģ��
	const string data_type	���ͼ���������ͣ�uchar char ushort short int float,Ĭ��ֵ��image_in������ͬ��
	�����
	cv::Mat& image_out		���ͼ��
	*/
void theKIIdogFilter(cv::Mat& image_in,cv::Mat& face_mask,cv::Mat& image_out,const string data_type);