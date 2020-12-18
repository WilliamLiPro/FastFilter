/***************************************************************
	>���ƣ������Cholesky�ֽ�
	>���ߣ���ά��
	>��ϵ��ʽ��williamli_pro@163.com
	>ʵ�ֶ������Գƾ����Cholesky�ֽ�
	>����Ҫ�㣺
	>1.��ͨ�����Cholesky�ֽ�
	>2.ϡ������Cholesky�ֽ�

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

struct Point_Value	//�����е�ɢ��
{
	int	row,col;	//�����е�����(��,��)
	float value;			//������Ӧֵ
};


/*	��ͨ�����Cholesky�ֽ�	
	���룺
	cv::Mat& symmetry_matrix		�����Գƾ���
	�����
	cv::Mat& down_triangle_matrix	�����Ǿ���
	����������falseʱ���������*/
bool normalCholeskyAnalyze(cv::Mat& symmetry_matrix,cv::Mat& down_triangle_matrix);

/*	ϡ������Cholesky�ֽ�	
	���룺
	vector<Point_Value>& sparse_matrix			�����Գƾ���(Ҫ��Ԫ�ذ�����˳������)
	�����
	vector<Point_Value>& down_triangle_matrix	�����Ǿ���
	����������falseʱ���������*/
bool sparseCholeskyAnalyze(vector<Point_Value>& sparse_matrix,vector<Point_Value>& down_triangle_matrix);

/*	Cholesky�ֽ���Ժ���	*/
bool choleskyAnalyzeTest();
