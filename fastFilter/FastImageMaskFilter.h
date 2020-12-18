/***************************************************************
	>类名：邻域（模板）运算
	>作者：李维鹏
	>联系方式：248636779@163.com
	>实现图像的行列对称模板与非对称模板运算
	>技术要点：
	>1.滤波模板分析(见filterAnalysis.h)
	>2.图像关于模板的相关运算
	>	2.1 2D模板滤波
	>	2.2 二维分离项滤波
	>	2.3 常数滤波	
	>	2.4 x,y向量模板滤波
	>	2.5 x,y对称向量模板滤波
	>	2.6 x,y等差向量滤波
	>3.图像非边缘滤波与边缘滤波
	>4.整形与float型图像的滤波

	备注:	(1)本程序基于多通道float型数据运算，支持各类输入图像
			(2)3与4技术要点融入了具体的计算程序

****************************************************************/

#pragma once

//windows界面库

// C++标准库
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>

// OpenCV
#include <opencv2/opencv.hpp>

//滤波器分解
#include "filterAnalysis.h"
//基本计算函数
#include "basicCalculate.h"
//Cholesky分解
#include "choleskyAnalyze.h"
//图像积分
#include "imageIntegration.h"

using namespace std;

//基于模板分解的快速滤波类
class FastImageMaskFilter
{
public:
	/*初始化函数*/
	FastImageMaskFilter();

	/*快速滤波函数*/
	bool runFastImageMaskFilter(cv::Mat& image_in,cv::Mat& image_out,const string data_out_type="",bool image_edge=true);

	/*模板输入函数*/
	bool inputMastFilter(cv::Mat& filter_mask,float gamma=0.05);

	float time_used[5];//各环节算法耗时(模板分析，整形化，图像边缘扩展，滤波)
	float standard_error_;						/*模板拆解均方标准差*/
private:

	//模板分解结果
	cv::Mat filter_mask_;						/*原始模板*/
	Vector_Mask filter_mask_x_,filter_mask_y_;	/*滤波模板x,y分量*/
	float mask_const_;							/*常数值*/
	vector<Point_Value> xy_delta_mask_;			/*分离量*/

	Int_Mask_Divide int_mask_;					/*整形化模板*/

	bool flag_divide_;								/*输出模板拆解形式（ture为可拆解，false为不可拆解）*/

};

//	*************************	快速滤波部分	**************************

/*	邻域模板运算(测试用函数)
	输入：
	cv::Mat& image_in			输入图像(单通道或多通道，任意类型)
	cv::Mat& filter_mask		与输入图像进行相关运算的模板(单通道,float型）
	float& gamma				模板分解信噪比阈值
	const string data_type			输出图像数据类型（uchar int float）
	int mask_divide_type		模板分解类型（0:不分解；1:xy分解；2:xy常数分解）
	bool filters_result_expand	模板运算结果是否扩展（true：扩展后输出矩阵与输入矩阵尺寸相同，否则按照模板大小扣除相应尺寸）
	输出：
	float& standard_error		模板分解均方标准差
	cv::Mat& image_out			输出图像(float或整型)

	当函数返回false时，程序出错*/
bool imageFiltering(cv::Mat& image_in,cv::Mat& filter_mask,float& gamma,const string data_type,int mask_divide_type,bool filters_result_expand,float& standard_error,cv::Mat& image_out);

/*	图像扩展函数
	输入：
	cv::Mat& image_in			输入图像(单通道或多通道，任意类型)
	cv::Vec2i& mask_size		模板尺寸		
	输出：
	cv::Mat& image_out			输出图像

	按照模板大小mask_size将image_in扩展至(rows_in+mask_size[0]-1,cols_in+mask_size[1]-1)
	并存入image_out*/
bool imageExtension(cv::Mat& image_in,cv::Vec2i& mask_size,cv::Mat& image_out);

/*	2.1 2D模板滤波
	输入：
	cv::Mat& image_in			输入图像(单通道或多通道，任意类型)
	cv::Mat& filter_mask		与输入图像进行相关运算的模板(单通道）
	const string data_type		输出图像数据类型（uchar char ushort short int float,默认与image_in类型相同）
	输出：
	cv::Mat& image_out			输出图像*/
bool tDmaskImageFiltering(cv::Mat& image_in,cv::Mat& filter_mask,const string data_type,cv::Mat& image_out);

/*	2.2 常数项滤波（模板函数）*/
/*	2.3 xy分离项滤波（模板函数）*/

/*	2.4 x,y向量模板滤波	
	输入：
	cv::Mat& image_in			输入图像(单通道或多通道，任意类型)
	Vector_Mask& x_filter		x模板向量
	Vector_Mask& y_filter		y模板向量
	Int_Mask_Divide& int_mask	整数模板
	const string data_type		输出图像数据类型（uchar char ushort short int float,默认与image_in类型相同）
	输出：
	cv::Mat& image_out			输出图像*/
bool vectorMaskImageFiltering(cv::Mat& image_in,Vector_Mask& x_filter,Vector_Mask& y_filter,Int_Mask_Divide& int_mask,const string data_type,cv::Mat& image_out);

/*	快速滤波测试程序	*/
bool fastImageFilterTest(const char* image_path);