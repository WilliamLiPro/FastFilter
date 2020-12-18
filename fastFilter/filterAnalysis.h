/***************************************************************
	>���������˲�ģ�����
	>���ߣ���ά��
	>��ϵ��ʽ��williamli_pro@163.com
	>ʵ�ֶ�1ά��2ά�˲����ķ�����ֽ�
	>����Ҫ�㣺
	>1.ģ�����
	>	1.1 ģ��Ļ���x,y�����ֽ�
	>	1.2 ����ģ���x,y��ֽ�
	>	1.3 ģ��Ļ���x,y,const�ֽ�
	>	1.4 ����ģ���x,y,const�ֽ�
	>	1.5 ����ģ��ĶԳ��Է���
	>	1.6 ���ƵȲ�����ģ��ķֽ�����

	��ע���������Ĭ��ģ����������Ϊfloat

****************************************************************/

#pragma once

// C++��׼��
#include <iostream>
#include <fstream>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

//�������㺯��
#include "basicCalculate.h"
//Cholesky�ֽ�
#include "choleskyAnalyze.h"
//ͼ�����
#include "imageIntegration.h"
//������Ŀ�꺯����С����Newton�������
#include "leastSquaresAndNewton.h"

using namespace std;

//	************************* �ṹ�� ***************************

//	ģ�������ṹ��
struct Vector_Mask
{
	vector<float> basic_vector;		/*	��������	*/
	int size;						/*	��������	*/
	int state;						/*	�����ֽ�״̬(0���ɷ֣�1�Գƣ�2���Կɷ�)	*/

	vector<float> symmetry_vector;	/*	�Գ�����	*/
	float filter_constant;			/*	ģ���������ĳ�����	*/
	float filter_linear;			/*	ģ����������һ����	*/
	float filter_p2;				/*	ģ���������Ķ�����	*/
	vector<cv::Vec2f> delta_mask;	/*	�ֽ�����	*/

	float standard_error;			/*	ģ��ֽ��׼��	*/
};

//	����ģ�������ṹ��
struct Int_Vector_Mask
{
	vector<int> basic_vector;		/*	��������	*/
	int size;						/*	��������	*/
	int state;						/*	�����ֽ�״̬(0���ɷ֣�1�Գƣ�2���Կɷ�)	*/

	vector<int> symmetry_vector;	/*	�Գ�����	*/
	int filter_constant;			/*	ģ���������ĳ�����	*/
	int filter_linear;				/*	ģ����������һ����	*/
	float filter_p2;				/*	ģ���������Ķ�����	*/
	vector<cv::Vec2i> delta_mask;	/*	�ֽ�����	*/

	int denominator;				/*	��һ����ĸ	*/

	float standard_error;			/*	ģ��ֽ��׼��	*/
};

//	���λ�ģ��ֽ�ṹ��
struct Int_Mask_Divide
{
	cv::Mat int_mask;					/*	����ģ��	*/
	cv::Vec2i size;						/*	ģ��ߴ�(����,���� or y,x)	*/
	bool state;							/*	ģ��ֽ�״̬*/
	int mask_denominator;				/*	ģ���ĸ	*/

	Int_Vector_Mask vector_x;			/*	x����ģ��	*/
	Int_Vector_Mask vector_y;			/*	y����ģ��	*/

	cv::Vec2i mask_const;				/*	������������ĸ	*/
	vector<cv::Vec4i> xy_delta_mask;	/*	�ֽ�ʣ����(ǰ��������Ϊ����x,y��������Ϊ���ӷ�ĸ)*/

	float standard_error;				/*	ģ��ֽ��׼��	*/
};

//	*************************	ģ���������	**************************

/*	ģ�����Version 1.0
	���룺
	cv::Mat& filter_mask				ģ�����
	float& gamma						�����
	�����
	Vector_Mask& filter_mask_x			x���򣨺�������ģ��
	Vector_Mask& filter_mask_y			y������������ģ��
	vector<Point_Value>& xy_delta_mask	xy�ֽ�����(x,y,df)
	float& standard_error				������׼��
	bool& flag_out						���ģ������ʽ��tureΪ�ɲ�⣬falseΪ���ɲ�⣩
	����������falseʱ���������*/
bool filterMaskAnalysis(cv::Mat& filter_mask,float& gamma,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,vector<Point_Value>& xy_delta_mask,float& standard_error,bool& flag_out);

/*	ģ�����Version 2.0.1	
	���룺
	cv::Mat& filter_mask				ģ�����
	float& gamma						�����
	�����
	Vector_Mask& filter_mask_x			x���򣨺�������ģ��
	Vector_Mask& filter_mask_y			y������������ģ��
	float& mask_const					ģ��ֽ�ĳ�����
	vector<Point_Value>& xy_delta_mask	xy�ֽ�����(x,y,df)
	float& standard_error				������׼��
	bool& flag_out						���ģ������ʽ��tureΪ�ɲ�⣬falseΪ���ɲ�⣩
	����������falseʱ���������*/
bool filterMaskAnalysis(cv::Mat& filter_mask,float& gamma,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,float& mask_const,vector<Point_Value>& xy_delta_mask,float& standard_error,bool& flag_out);			//��������ʽ����

/*	1.1 ģ��Ļ���x,y��ֽ�	
	���룺
	cv::Mat& filter_mask				ģ�����
	�����
	vector<float>& filter_mask_x		x���򣨺�������ģ��
	vector<float>& filter_mask_y		y������������ģ��
	float& standard_error				������׼��
	bool flag_out						����ģ���Ƿ�ɲ�⣨tureΪ�ɲ�⣬��ʱģ���x,y���������Ч��
	����������falseʱ���������*/
bool basicFilterMaskXYAnalysis(cv::Mat& filter_mask,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& standard_error,bool& flag_out);

/*	1.2 ����ģ���x,y��ֽ�	
	���룺
	cv::Mat& filter_mask				ģ�����
	float& gamma						�����
	�����
	vector<float>& filter_mask_x		x���򣨺�������ģ��
	vector<float>& filter_mask_y		y������������ģ��
	vector<Point_Value>& delta_mask		ϡ��������(x,y,df)
	float& standard_error				ģ��ֽ�ľ�����׼��
	bool flag_out						����ģ���Ƿ�ɲ�⣨tureΪ�ɲ�⣬��ʱģ���x,y���������Ч��
	����������falseʱ���������*/
bool complexFilterMaskXYAnalysis(cv::Mat& filter_mask,float& gamma,vector<float>& filter_mask_x,vector<float>& filter_mask_y,vector<Point_Value>& delta_mask,float& standard_error,bool& flag_out);

/*	1.3 ģ��Ļ���x,y,const�ֽ�	
	���룺
	cv::Mat& filter_mask				ģ�����
	�����
	vector<float>& filter_mask_x		x���򣨺�������ģ��
	vector<float>& filter_mask_y		y������������ģ��
	float& mask_const					ģ��ֽ�ĳ�����
	float& standard_error				������׼��
	bool flag_out						����ģ���Ƿ�ɲ�⣨tureΪ�ɲ�⣬��ʱģ���x,y���������Ч��
	����������falseʱ���������*/
bool basicFilterMaskXYconstAnalysis(cv::Mat& filter_mask,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& mask_const,float& standard_error,bool& flag_out);

/*	1.4 ����ģ���x,y,const�ֽ�	
	���룺
	cv::Mat& filter_mask				ģ�����
	float& gamma						�����
	�����
	vector<float>& filter_mask_x		x���򣨺�������ģ��
	vector<float>& filter_mask_y		y������������ģ��
	float& mask_const					ģ��ֽ�ĳ�����
	vector<Point_Value>& delta_mask		ϡ��������(x,y,df)
	float& standard_error				ģ��ֽ�ľ�����׼��
	bool flag_out						����ģ���Ƿ�ɲ�⣨tureΪ�ɲ�⣬��ʱģ���x,y���������Ч��
	����������falseʱ���������*/
bool complexFilterMaskXYconstAnalysis(cv::Mat& filter_mask,float& gamma,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& mask_const,vector<Point_Value>& delta_mask,float& standard_error,bool& flag_out);


/*	1.5 ����ģ��ĶԳ��Է���
	���룺
	vector<float>& filter_mask		ģ������
	float& gamma					�����
	�����
	vector<float>& symmetry_mask	�Գ�ģ��
	vector<cv::Vec2f>& delta_mask	ϡ���������(x,df)
	float& standard_error			ģ��ֽ�ľ�����׼��
	bool flag_out					����ģ���Ƿ�ɲ�⣨tureΪ�ɲ�⣬��ʱģ��ĶԳƻ������Ч��
	����������falseʱ���������*/
bool vectorFilterMaskSymmetryAnalysis(vector<float>& filter_mask,float& gamma,vector<float>& symmetry_mask,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out);


/*	1.6 ���ƵȲ�����ģ��ķֽ�
	���룺
	vector<float>& filter_mask		ģ������
	float& gamma					�����
	�����
	float& filter_constant			ģ���������ĳ�����
	float& filter_linear			ģ����������һ����
	vector<cv::Vec2f>& delta_mask	ϡ���������(x,df)
	float& standard_error			ģ��ֽ�ľ�����׼��
	bool flag_out					����ģ���Ƿ�ɲ�⣨tureΪ�ɲ�⣬��ʱģ���x,y���������Ч��
	����������falseʱ���������*/
bool gradeVectorFilterMaskAnalysisOriginal(vector<float>& filter_mask,float& gamma,float& filter_constant,float& filter_linear,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out);
bool gradeVectorFilterMaskAnalysis(vector<float>& filter_mask,float& gamma,float& filter_constant,float& filter_linear,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out);
void sumSqrtFunc(cv::Mat& base_para,cv::Mat& fun_in,float& fun_out);

/*	1.7 ����ģ��Ķ��ηֽ�
	���룺
	vector<float>& filter_mask		ģ������
	float& gamma					�����
	�����
	float& filter_constant			ģ���������ĳ�����
	float& filter_linear			ģ����������һ����
	float& filter_p2				ģ���������Ķ�����
	vector<cv::Vec2f>& delta_mask	ϡ���������(x,df)
	float& standard_error			ģ��ֽ�ľ�����׼��
	bool flag_out					����ģ���Ƿ�ɲ�⣨tureΪ�ɲ�⣬��ʱģ���x,y���������Ч��
	����������falseʱ���������*/
bool parabolaVectorFilterMaskAnalysis(vector<float>& filter_mask,float& gamma,float& filter_constant,float& filter_linear,float& filter_p2,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out);
void sumSqrtFunc2(cv::Mat& base_para,cv::Mat& fun_in,float& fun_out);


/*	1.8 ģ������
	���룺
	float gamma							�����
	cv::Mat& filter_mask				ģ�����
	bool& flag_out						ģ���Ƿ�ɲ��
	Vector_Mask& filter_mask_x			x���򣨺�������ģ��
	Vector_Mask& filter_mask_y			y������������ģ��
	float& mask_const					ģ��ֽ�ĳ�����
	vector<Point_Value>& xy_delta_mask	xy�ֽ�����(x,y,df)
	float& standard_error				������׼��

	�����
	Int_Mask_Divide& int_mask			���λ�ģ��
	float& standard_error				ģ��ֽ�ľ�����׼��
	����������falseʱ���������*/
bool integerMask(float gamma,cv::Mat& filter_mask,bool& flag_out,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,float& mask_const,vector<Point_Value>& xy_delta_mask,float& standard_error,Int_Mask_Divide& int_mask);

/*	ģ��������Ժ���*/
bool filterMaskTest();
