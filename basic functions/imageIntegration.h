/***************************************************************
	>���ƣ�ͼ��Ļ�������
	>���ߣ���ά��
	>��ϵ��ʽ��williamli_pro@163.com
	>ʵ�ֶ�ͼ��Ŀ�����������
	>����Ҫ�㣺
	>1.uchar�Ͷ�ͨ��ͼ��Ŀ�����������
	>2.һ���ͨ��ͼ��Ŀ�����������

****************************************************************/

#pragma once

// C++��׼��
#include <iostream>
#include <fstream>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

using namespace std;

//	************************* ��д������ͷ�ļ� ***************************


/*	uchar(char)�Ͷ�ͨ��ͼ��Ŀ�����������
	���룺
	cv::Mat& image_in		����ͼ��uchar(char)�ͣ�
	int data_type			�������ͣ�0 uchar,1 char��
	�����
	cv::Mat& image_out		���ͼ��int�ͣ�
	����������falseʱ���������*/
bool fastUcharImageIntegration(cv::Mat& image_in,int data_type,cv::Mat& image_out);

/*	һ���ͨ��ͼ��Ŀ�����������	
	���룺
	cv::Mat& image_in		����ͼ���������ͣ�
	�����
	cv::Mat& image_out		���ͼ��float�ͣ�
	����������falseʱ���������*/
bool fastNormalImageIntegration(cv::Mat& image_in,cv::Mat& image_out);

/*	ͼ�����������Ժ���	*/
bool imageIntegrationTest();
