/***************************************************************
	>���ƣ��������㺯����
	>���ߣ���ά��
	>��ϵ��ʽ��williamli_pro@163.com
	>ʵ��һЩ������������
	>Ҫ�㣺
	>1.��������ֵ����Ȩ��ֵ������
	>2.�������������Լ��
	>3.ʵ������������ֽ�

****************************************************************/

#pragma once

#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std;
//***********	1.��������ֵ����Ȩ��ֵ������	*************
//����������ֵ
inline float vectorMean(vector<float>& input_vec);

//����������Ȩ��ֵ
inline float vectorMean(vector<float>& input_vec,vector<float>& weight_vec);

//����ָ����ֵ������������
void vectorStd(vector<float>& input_vec,float& vec_aver,float& vec_std);

//***********	2.�������������Լ��	*************
int calcuGCD(int a,int b);

//***********	3.ʵ�������������������ֽ�	*************
/*	����շת�����
	float gamma:			�������
	cv::Mat& input_mat��	�������
	cv::Mat& output_mat��	������ξ���
	int& denominator:		����ķ�ĸ
*/
void computeRationalMatrix(float gamma,cv::Mat& input_mat,cv::Mat& output_mat,int& denominator);
/*	����շת�����
	float gamma:					�������
	vector<float>& input_vector��	��������
	vector<int>& output_vector�������������
	int& denominator:			�����ķ�ĸ
*/
void computeRationalVector(float gamma,vector<float>& input_vector,vector<int>& output_vector,int& denominator);
/*	ʵ������������շת�����
	float gamma:				�������
	float& input_number��		������
	int& output_numerator��		�������
	int& output_denominator��	�����ĸ
*/
void realNumRational(float& gamma,float& input_number,int& output_numerator,int& output_denominator);

