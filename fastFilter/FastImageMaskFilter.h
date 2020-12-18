/***************************************************************
	>����������ģ�壩����
	>���ߣ���ά��
	>��ϵ��ʽ��248636779@163.com
	>ʵ��ͼ������жԳ�ģ����ǶԳ�ģ������
	>����Ҫ�㣺
	>1.�˲�ģ�����(��filterAnalysis.h)
	>2.ͼ�����ģ����������
	>	2.1 2Dģ���˲�
	>	2.2 ��ά�������˲�
	>	2.3 �����˲�	
	>	2.4 x,y����ģ���˲�
	>	2.5 x,y�Գ�����ģ���˲�
	>	2.6 x,y�Ȳ������˲�
	>3.ͼ��Ǳ�Ե�˲����Ե�˲�
	>4.������float��ͼ����˲�

	��ע:	(1)��������ڶ�ͨ��float���������㣬֧�ָ�������ͼ��
			(2)3��4����Ҫ�������˾���ļ������

****************************************************************/

#pragma once

//windows�����

// C++��׼��
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>

// OpenCV
#include <opencv2/opencv.hpp>

//�˲����ֽ�
#include "filterAnalysis.h"
//�������㺯��
#include "basicCalculate.h"
//Cholesky�ֽ�
#include "choleskyAnalyze.h"
//ͼ�����
#include "imageIntegration.h"

using namespace std;

//����ģ��ֽ�Ŀ����˲���
class FastImageMaskFilter
{
public:
	/*��ʼ������*/
	FastImageMaskFilter();

	/*�����˲�����*/
	bool runFastImageMaskFilter(cv::Mat& image_in,cv::Mat& image_out,const string data_out_type="",bool image_edge=true);

	/*ģ�����뺯��*/
	bool inputMastFilter(cv::Mat& filter_mask,float gamma=0.05);

	float time_used[5];//�������㷨��ʱ(ģ����������λ���ͼ���Ե��չ���˲�)
	float standard_error_;						/*ģ���������׼��*/
private:

	//ģ��ֽ���
	cv::Mat filter_mask_;						/*ԭʼģ��*/
	Vector_Mask filter_mask_x_,filter_mask_y_;	/*�˲�ģ��x,y����*/
	float mask_const_;							/*����ֵ*/
	vector<Point_Value> xy_delta_mask_;			/*������*/

	Int_Mask_Divide int_mask_;					/*���λ�ģ��*/

	bool flag_divide_;								/*���ģ������ʽ��tureΪ�ɲ�⣬falseΪ���ɲ�⣩*/

};

//	*************************	�����˲�����	**************************

/*	����ģ������(�����ú���)
	���룺
	cv::Mat& image_in			����ͼ��(��ͨ�����ͨ������������)
	cv::Mat& filter_mask		������ͼ�������������ģ��(��ͨ��,float�ͣ�
	float& gamma				ģ��ֽ��������ֵ
	const string data_type			���ͼ���������ͣ�uchar int float��
	int mask_divide_type		ģ��ֽ����ͣ�0:���ֽ⣻1:xy�ֽ⣻2:xy�����ֽ⣩
	bool filters_result_expand	ģ���������Ƿ���չ��true����չ������������������ߴ���ͬ��������ģ���С�۳���Ӧ�ߴ磩
	�����
	float& standard_error		ģ��ֽ������׼��
	cv::Mat& image_out			���ͼ��(float������)

	����������falseʱ���������*/
bool imageFiltering(cv::Mat& image_in,cv::Mat& filter_mask,float& gamma,const string data_type,int mask_divide_type,bool filters_result_expand,float& standard_error,cv::Mat& image_out);

/*	ͼ����չ����
	���룺
	cv::Mat& image_in			����ͼ��(��ͨ�����ͨ������������)
	cv::Vec2i& mask_size		ģ��ߴ�		
	�����
	cv::Mat& image_out			���ͼ��

	����ģ���Сmask_size��image_in��չ��(rows_in+mask_size[0]-1,cols_in+mask_size[1]-1)
	������image_out*/
bool imageExtension(cv::Mat& image_in,cv::Vec2i& mask_size,cv::Mat& image_out);

/*	2.1 2Dģ���˲�
	���룺
	cv::Mat& image_in			����ͼ��(��ͨ�����ͨ������������)
	cv::Mat& filter_mask		������ͼ�������������ģ��(��ͨ����
	const string data_type		���ͼ���������ͣ�uchar char ushort short int float,Ĭ����image_in������ͬ��
	�����
	cv::Mat& image_out			���ͼ��*/
bool tDmaskImageFiltering(cv::Mat& image_in,cv::Mat& filter_mask,const string data_type,cv::Mat& image_out);

/*	2.2 �������˲���ģ�庯����*/
/*	2.3 xy�������˲���ģ�庯����*/

/*	2.4 x,y����ģ���˲�	
	���룺
	cv::Mat& image_in			����ͼ��(��ͨ�����ͨ������������)
	Vector_Mask& x_filter		xģ������
	Vector_Mask& y_filter		yģ������
	Int_Mask_Divide& int_mask	����ģ��
	const string data_type		���ͼ���������ͣ�uchar char ushort short int float,Ĭ����image_in������ͬ��
	�����
	cv::Mat& image_out			���ͼ��*/
bool vectorMaskImageFiltering(cv::Mat& image_in,Vector_Mask& x_filter,Vector_Mask& y_filter,Int_Mask_Divide& int_mask,const string data_type,cv::Mat& image_out);

/*	�����˲����Գ���	*/
bool fastImageFilterTest(const char* image_path);