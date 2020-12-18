/***************************************************************
	>名称：基本计算函数集
	>作者：李维鹏
	>联系方式：williamli_pro@163.com
	>实现一些基本函数运算
	>要点：
	>1.求向量均值、加权均值、方差
	>2.求两个数的最大公约数
	>3.实数矩阵的有理化分解

****************************************************************/

#pragma once

#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std;
//***********	1.求向量均值、加权均值、方差	*************
//计算向量均值
inline float vectorMean(vector<float>& input_vec);

//计算向量加权均值
inline float vectorMean(vector<float>& input_vec,vector<float>& weight_vec);

//根据指定均值计算向量方差
void vectorStd(vector<float>& input_vec,float& vec_aver,float& vec_std);

//***********	2.求两个数的最大公约数	*************
int calcuGCD(int a,int b);

//***********	3.实数矩阵（向量）的有理化分解	*************
/*	采用辗转相除法
	float gamma:			允许误差
	cv::Mat& input_mat：	输入矩阵
	cv::Mat& output_mat：	输出整形矩阵
	int& denominator:		矩阵的分母
*/
void computeRationalMatrix(float gamma,cv::Mat& input_mat,cv::Mat& output_mat,int& denominator);
/*	采用辗转相除法
	float gamma:					允许误差
	vector<float>& input_vector：	输入向量
	vector<int>& output_vector：输出整形向量
	int& denominator:			向量的分母
*/
void computeRationalVector(float gamma,vector<float>& input_vector,vector<int>& output_vector,int& denominator);
/*	实数有理化：采用辗转相除法
	float gamma:				允许误差
	float& input_number：		输入数
	int& output_numerator：		输出分子
	int& output_denominator：	输出分母
*/
void realNumRational(float& gamma,float& input_number,int& output_numerator,int& output_denominator);

