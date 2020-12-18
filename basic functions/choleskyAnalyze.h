/***************************************************************
	>名称：矩阵的Cholesky分解
	>作者：李维鹏
	>联系方式：williamli_pro@163.com
	>实现对正定对称矩阵的Cholesky分解
	>技术要点：
	>1.普通矩阵的Cholesky分解
	>2.稀疏矩阵的Cholesky分解

****************************************************************/

#pragma once

// C++标准库
#include <iostream>
#include <fstream>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

using namespace std;

//	************************* 编写函数的头文件 ***************************

struct Point_Value	//矩阵中的散点
{
	int	row,col;	//矩阵中的坐标(行,列)
	float value;			//坐标点对应值
};


/*	普通矩阵的Cholesky分解	
	输入：
	cv::Mat& symmetry_matrix		正定对称矩阵
	输出：
	cv::Mat& down_triangle_matrix	下三角矩阵
	当函数返回false时，程序出错*/
bool normalCholeskyAnalyze(cv::Mat& symmetry_matrix,cv::Mat& down_triangle_matrix);

/*	稀疏矩阵的Cholesky分解	
	输入：
	vector<Point_Value>& sparse_matrix			正定对称矩阵(要求元素按行列顺序排列)
	输出：
	vector<Point_Value>& down_triangle_matrix	下三角矩阵
	当函数返回false时，程序出错*/
bool sparseCholeskyAnalyze(vector<Point_Value>& sparse_matrix,vector<Point_Value>& down_triangle_matrix);

/*	Cholesky分解测试函数	*/
bool choleskyAnalyzeTest();
