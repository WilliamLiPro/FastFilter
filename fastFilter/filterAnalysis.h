/***************************************************************
	>程序名：滤波模板分析
	>作者：李维鹏
	>联系方式：williamli_pro@163.com
	>实现对1维与2维滤波器的分析与分解
	>技术要点：
	>1.模板分析
	>	1.1 模板的基本x,y向量分解
	>	1.2 复杂模板的x,y向分解
	>	1.3 模板的基本x,y,const分解
	>	1.4 复杂模板的x,y,const分解
	>	1.5 向量模板的对称性分析
	>	1.6 近似等差向量模板的分解运算

	备注：本程序基默认模板数据类型为float

****************************************************************/

#pragma once

// C++标准库
#include <iostream>
#include <fstream>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

//基本计算函数
#include "basicCalculate.h"
//Cholesky分解
#include "choleskyAnalyze.h"
//图像积分
#include "imageIntegration.h"
//非线性目标函数最小化的Newton迭代求解
#include "leastSquaresAndNewton.h"

using namespace std;

//	************************* 结构体 ***************************

//	模板向量结构体
struct Vector_Mask
{
	vector<float> basic_vector;		/*	基本向量	*/
	int size;						/*	向量长度	*/
	int state;						/*	向量分解状态(0不可分；1对称；2线性可分)	*/

	vector<float> symmetry_vector;	/*	对称向量	*/
	float filter_constant;			/*	模板关于坐标的常数项	*/
	float filter_linear;			/*	模板关于坐标的一次项	*/
	float filter_p2;				/*	模板关于坐标的二次项	*/
	vector<cv::Vec2f> delta_mask;	/*	分解余项	*/

	float standard_error;			/*	模板分解标准差	*/
};

//	整形模板向量结构体
struct Int_Vector_Mask
{
	vector<int> basic_vector;		/*	基本向量	*/
	int size;						/*	向量长度	*/
	int state;						/*	向量分解状态(0不可分；1对称；2线性可分)	*/

	vector<int> symmetry_vector;	/*	对称向量	*/
	int filter_constant;			/*	模板关于坐标的常数项	*/
	int filter_linear;				/*	模板关于坐标的一次项	*/
	float filter_p2;				/*	模板关于坐标的二次项	*/
	vector<cv::Vec2i> delta_mask;	/*	分解余项	*/

	int denominator;				/*	归一化分母	*/

	float standard_error;			/*	模板分解标准差	*/
};

//	整形化模板分解结构体
struct Int_Mask_Divide
{
	cv::Mat int_mask;					/*	整形模板	*/
	cv::Vec2i size;						/*	模板尺寸(行数,列数 or y,x)	*/
	bool state;							/*	模板分解状态*/
	int mask_denominator;				/*	模板分母	*/

	Int_Vector_Mask vector_x;			/*	x方向模板	*/
	Int_Vector_Mask vector_y;			/*	y方向模板	*/

	cv::Vec2i mask_const;				/*	常数项分子与分母	*/
	vector<cv::Vec4i> xy_delta_mask;	/*	分解剩余项(前两项依次为坐标x,y，后两项为分子分母)*/

	float standard_error;				/*	模板分解标准差	*/
};

//	*************************	模板分析部分	**************************

/*	模板分析Version 1.0
	输入：
	cv::Mat& filter_mask				模板矩阵
	float& gamma						信噪比
	输出：
	Vector_Mask& filter_mask_x			x方向（横向）向量模板
	Vector_Mask& filter_mask_y			y方向（纵向）向量模板
	vector<Point_Value>& xy_delta_mask	xy分解余项(x,y,df)
	float& standard_error				均方标准差
	bool& flag_out						输出模板拆解形式（ture为可拆解，false为不可拆解）
	当函数返回false时，程序出错*/
bool filterMaskAnalysis(cv::Mat& filter_mask,float& gamma,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,vector<Point_Value>& xy_delta_mask,float& standard_error,bool& flag_out);

/*	模板分析Version 2.0.1	
	输入：
	cv::Mat& filter_mask				模板矩阵
	float& gamma						信噪比
	输出：
	Vector_Mask& filter_mask_x			x方向（横向）向量模板
	Vector_Mask& filter_mask_y			y方向（纵向）向量模板
	float& mask_const					模板分解的常数项
	vector<Point_Value>& xy_delta_mask	xy分解余项(x,y,df)
	float& standard_error				均方标准差
	bool& flag_out						输出模板拆解形式（ture为可拆解，false为不可拆解）
	当函数返回false时，程序出错*/
bool filterMaskAnalysis(cv::Mat& filter_mask,float& gamma,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,float& mask_const,vector<Point_Value>& xy_delta_mask,float& standard_error,bool& flag_out);			//浮点数形式向量

/*	1.1 模板的基本x,y向分解	
	输入：
	cv::Mat& filter_mask				模板矩阵
	输出：
	vector<float>& filter_mask_x		x方向（横向）向量模板
	vector<float>& filter_mask_y		y方向（纵向）向量模板
	float& standard_error				均方标准差
	bool flag_out						输入模板是否可拆解（ture为可拆解，此时模板的x,y分量拆解有效）
	当函数返回false时，程序出错*/
bool basicFilterMaskXYAnalysis(cv::Mat& filter_mask,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& standard_error,bool& flag_out);

/*	1.2 复杂模板的x,y向分解	
	输入：
	cv::Mat& filter_mask				模板矩阵
	float& gamma						信噪比
	输出：
	vector<float>& filter_mask_x		x方向（横向）向量模板
	vector<float>& filter_mask_y		y方向（纵向）向量模板
	vector<Point_Value>& delta_mask		稀疏误差矩阵(x,y,df)
	float& standard_error				模板分解的均方标准差
	bool flag_out						输入模板是否可拆解（ture为可拆解，此时模板的x,y分量拆解有效）
	当函数返回false时，程序出错*/
bool complexFilterMaskXYAnalysis(cv::Mat& filter_mask,float& gamma,vector<float>& filter_mask_x,vector<float>& filter_mask_y,vector<Point_Value>& delta_mask,float& standard_error,bool& flag_out);

/*	1.3 模板的基本x,y,const分解	
	输入：
	cv::Mat& filter_mask				模板矩阵
	输出：
	vector<float>& filter_mask_x		x方向（横向）向量模板
	vector<float>& filter_mask_y		y方向（纵向）向量模板
	float& mask_const					模板分解的常数项
	float& standard_error				均方标准差
	bool flag_out						输入模板是否可拆解（ture为可拆解，此时模板的x,y分量拆解有效）
	当函数返回false时，程序出错*/
bool basicFilterMaskXYconstAnalysis(cv::Mat& filter_mask,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& mask_const,float& standard_error,bool& flag_out);

/*	1.4 复杂模板的x,y,const分解	
	输入：
	cv::Mat& filter_mask				模板矩阵
	float& gamma						信噪比
	输出：
	vector<float>& filter_mask_x		x方向（横向）向量模板
	vector<float>& filter_mask_y		y方向（纵向）向量模板
	float& mask_const					模板分解的常数项
	vector<Point_Value>& delta_mask		稀疏误差矩阵(x,y,df)
	float& standard_error				模板分解的均方标准差
	bool flag_out						输入模板是否可拆解（ture为可拆解，此时模板的x,y分量拆解有效）
	当函数返回false时，程序出错*/
bool complexFilterMaskXYconstAnalysis(cv::Mat& filter_mask,float& gamma,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& mask_const,vector<Point_Value>& delta_mask,float& standard_error,bool& flag_out);


/*	1.5 向量模板的对称性分析
	输入：
	vector<float>& filter_mask		模板向量
	float& gamma					信噪比
	输出：
	vector<float>& symmetry_mask	对称模板
	vector<cv::Vec2f>& delta_mask	稀疏误差向量(x,df)
	float& standard_error			模板分解的均方标准差
	bool flag_out					输入模板是否可拆解（ture为可拆解，此时模板的对称化拆解有效）
	当函数返回false时，程序出错*/
bool vectorFilterMaskSymmetryAnalysis(vector<float>& filter_mask,float& gamma,vector<float>& symmetry_mask,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out);


/*	1.6 近似等差向量模板的分解
	输入：
	vector<float>& filter_mask		模板向量
	float& gamma					信噪比
	输出：
	float& filter_constant			模板关于坐标的常数项
	float& filter_linear			模板关于坐标的一次项
	vector<cv::Vec2f>& delta_mask	稀疏误差向量(x,df)
	float& standard_error			模板分解的均方标准差
	bool flag_out					输入模板是否可拆解（ture为可拆解，此时模板的x,y分量拆解有效）
	当函数返回false时，程序出错*/
bool gradeVectorFilterMaskAnalysisOriginal(vector<float>& filter_mask,float& gamma,float& filter_constant,float& filter_linear,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out);
bool gradeVectorFilterMaskAnalysis(vector<float>& filter_mask,float& gamma,float& filter_constant,float& filter_linear,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out);
void sumSqrtFunc(cv::Mat& base_para,cv::Mat& fun_in,float& fun_out);

/*	1.7 向量模板的二次分解
	输入：
	vector<float>& filter_mask		模板向量
	float& gamma					信噪比
	输出：
	float& filter_constant			模板关于坐标的常数项
	float& filter_linear			模板关于坐标的一次项
	float& filter_p2				模板关于坐标的二次项
	vector<cv::Vec2f>& delta_mask	稀疏误差向量(x,df)
	float& standard_error			模板分解的均方标准差
	bool flag_out					输入模板是否可拆解（ture为可拆解，此时模板的x,y分量拆解有效）
	当函数返回false时，程序出错*/
bool parabolaVectorFilterMaskAnalysis(vector<float>& filter_mask,float& gamma,float& filter_constant,float& filter_linear,float& filter_p2,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out);
void sumSqrtFunc2(cv::Mat& base_para,cv::Mat& fun_in,float& fun_out);


/*	1.8 模板有理化
	输入：
	float gamma							信噪比
	cv::Mat& filter_mask				模板矩阵
	bool& flag_out						模板是否可拆解
	Vector_Mask& filter_mask_x			x方向（横向）向量模板
	Vector_Mask& filter_mask_y			y方向（纵向）向量模板
	float& mask_const					模板分解的常数项
	vector<Point_Value>& xy_delta_mask	xy分解余项(x,y,df)
	float& standard_error				均方标准差

	输出：
	Int_Mask_Divide& int_mask			整形化模板
	float& standard_error				模板分解的均方标准差
	当函数返回false时，程序出错*/
bool integerMask(float gamma,cv::Mat& filter_mask,bool& flag_out,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,float& mask_const,vector<Point_Value>& xy_delta_mask,float& standard_error,Int_Mask_Divide& int_mask);

/*	模板分析测试函数*/
bool filterMaskTest();
