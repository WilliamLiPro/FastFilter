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

#include "theKIIfilter.h"

/*******************  KII������������  *******************/
template<typename T_p1,typename T_p2> void theKIIgaussFilterBasic(cv::Mat& image_in,cv::Mat& image_out,int n_kernelx,int n_kernely,T_p1 std_kernel);
template<typename T_p1,typename T_p2> void theKIIsobelFilter(cv::Mat& image_in,cv::Mat& image_out,int n_kernelx,int n_kernely,T_p1 std_kernel,T_p1 dy_kernel);


/*********************  KII�˲�����  ********************/

/*	��˹��KII�˲�����
	���룺
	cv::Mat& image_in		����ͼ��
	cv::Mat& filte_kernel	�˲���
	const string data_type	���ͼ���������ͣ�uchar char ushort short int float,Ĭ��ֵ��image_in������ͬ��
	�����
	cv::Mat& image_out		���ͼ��
	*/
void theKIIgaussFilter(cv::Mat& image_in,cv::Mat& filte_kernel,cv::Mat& image_out,const string data_type)
{

}

/*	sobel��KII�˲�����
	���룺
	cv::Mat& image_in		����ͼ��
	cv::Mat& filte_kernel	�˲���
	const string data_type	���ͼ���������ͣ�uchar char ushort short int float,Ĭ��ֵ��image_in������ͬ��
	�����
	cv::Mat& image_out		���ͼ��
	*/
void theKIIsobelFilter(cv::Mat& image_in,cv::Mat& filte_kernel,cv::Mat& image_out,const string data_type)
{

}

/*	DOG��KII�˲�����
	���룺
	cv::Mat& image_in		����ͼ��
	cv::Mat& filte_kernel	�˲���
	const string data_type	���ͼ���������ͣ�uchar char ushort short int float,Ĭ��ֵ��image_in������ͬ��
	�����
	cv::Mat& image_out		���ͼ��
	*/
void theKIIdogFilter(cv::Mat& image_in,cv::Mat& filte_kernel,cv::Mat& image_out,const string data_type)
{

}

/*	��׼����ģ��KII�˲�����
	���룺
	cv::Mat& image_in		����ͼ��
	cv::Mat& face_mask		��׼����ģ��
	const string data_type	���ͼ���������ͣ�uchar char ushort short int float,Ĭ��ֵ��image_in������ͬ��
	�����
	cv::Mat& image_out		���ͼ��
	*/
void theKIIface(cv::Mat& image_in,cv::Mat& face_mask,cv::Mat& image_out,const string data_type)
{

}