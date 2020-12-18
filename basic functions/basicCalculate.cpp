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

#include "basicCalculate.h"

//1.��������ֵ����Ȩ��ֵ������
//����������ֵ
inline float vectorMean(vector<float>& input_vec)
{
	int vec_l=input_vec.size();

	float vec_aver=0;
	for(int i=0;i<vec_l;i++)
	{
		vec_aver+=input_vec[i];
	}

	vec_aver=vec_aver/vec_l;

	return vec_aver;
}

//����������Ȩ��ֵ
inline float vectorMean(vector<float>& input_vec,vector<float>& weight_vec)
{
	int vec_l=input_vec.size();
	int weight_l=weight_vec.size();

	//Ȩ�غ�
	float sum_weight=0;
	for(int i=0;i<weight_l;i++)
		sum_weight+=weight_vec[i];

	//��ֵ
	float vec_aver=0;
	if(weight_l<vec_l)
	{
		for(int i=0;i<weight_l;i++)
			vec_aver+=input_vec[i]*weight_vec[i];

		vec_aver=vec_aver/weight_l;
	}
	else
	{
		for(int i=0;i<vec_l;i++)
			vec_aver+=input_vec[i];

		vec_aver=vec_aver/vec_l;
	}
	
	return vec_aver;
}

//����ָ����ֵ������������
void vectorStd(vector<float>& input_vec,float& vec_aver,float& vec_std)
{
	int vec_l=input_vec.size();

	vec_std=0;
	for(int i=0;i<vec_l;i++)
	{
		float d_element=input_vec[i]-vec_aver;
		vec_std+=d_element*d_element;
	}

	vec_std=sqrt(vec_std/vec_l);
}


//2.�������������Լ��
int calcuGCD(int a,int b)
{
	if(a==0||b==0)
	{
		return 1;
	}

	while(1)
	{
		int abs_a=abs(a);
		int abs_b=abs(b);

		int max_num=a;
		int min_num=b;

		if(abs_a<abs_b)
		{
			max_num=b;
			min_num=a;
		}

		int remainder=max_num%min_num;
		if(remainder==0)
		{
			return abs(min_num);
		}

		int divide_num=max_num/min_num;
		a=min_num;
		b=max_num-divide_num*min_num;
	}
}

//3.ʵ������������ֽ�
void computeRationalMatrix(float gamma,cv::Mat& input_mat,cv::Mat& output_mat,int& denominator)
{
	//ͳ����������ͳһ��������
	int mat_rows=input_mat.rows;
	int mat_cols=input_mat.cols;

	if(input_mat.depth()!=CV_32FC1)
		input_mat.convertTo(input_mat,CV_32F);

	//����Ԫ�ؾ���
	float* input_mat_ptr;
	float aver_square=0;
	for(int i=0;i<mat_rows;i++)
	{
		input_mat_ptr=input_mat.ptr<float>(i);
		for(int j=0;j<mat_cols;j++)
		{
			aver_square+=input_mat_ptr[j]*input_mat_ptr[j];
		}
	}
	aver_square=sqrt(aver_square/mat_rows/mat_cols);
	float thresold=gamma*aver_square;

	//������С����
	float remain_num=1;

	for(int i=0;i<mat_rows;i++)
	{
		input_mat_ptr=input_mat.ptr<float>(i);
		for(int j=0;j<mat_cols;j++)
		{
			if(abs(input_mat_ptr[j])>0.01*aver_square&&abs(input_mat_ptr[j])<remain_num)
				remain_num=abs(input_mat_ptr[j]);
		}
	}
	if(remain_num>1)
		remain_num=1;

	//�����ʼֵ
	cv::Mat output_mat_r=cv::Mat(mat_rows,mat_cols,CV_32SC1);
	output_mat=cv::Mat(mat_rows,mat_cols,CV_32SC1);
	int* output_mat_ptr,*output_mat_r_ptr;

	for(int i=0;i<mat_rows;i++)
	{
		input_mat_ptr=input_mat.ptr<float>(i);
		output_mat_ptr=output_mat.ptr<int>(i);

		for(int j=0;j<mat_cols;j++)
		{
			if(input_mat_ptr[j]>=0)
				output_mat_ptr[j]=input_mat_ptr[j]/remain_num+0.5;
			else
				output_mat_ptr[j]=input_mat_ptr[j]/remain_num-0.5;
		}
	}

	//ѭ������
	int max_out_num=0;//������

	for(int iterate=0;iterate<10;iterate++)
	{
		float min_num=remain_num;	//���ڼ��������
		remain_num=0;

		//�����ĸ
		denominator=1/min_num+0.5;

		//������Ӿ���
		for(int i=0;i<mat_rows;i++)
		{
			input_mat_ptr=input_mat.ptr<float>(i);
			output_mat_r_ptr=output_mat_r.ptr<int>(i);
			for(int j=0;j<mat_cols;j++)
			{
				if(input_mat_ptr[j]>=0)
					output_mat_r_ptr[j]=input_mat_ptr[j]/min_num+0.5;
				else
					output_mat_r_ptr[j]=input_mat_ptr[j]/min_num-0.5;

				if(abs(output_mat_r_ptr[j])>max_out_num)
					max_out_num=abs(output_mat_r_ptr[j]);

				float remainder=abs(float(output_mat_r_ptr[j])/denominator-input_mat_ptr[j]);//����
				if(remainder>remain_num)
					remain_num=remainder;
			}
		}

		if(max_out_num>127)//������󲻳���127
		{
			break;
		}

		output_mat_r.copyTo(output_mat);

		if(remain_num<thresold)//������Сֵ��С����ֵ�����������
			break;
	}

	//�����ĸ
	float input_mat_sum=0;
	int output_mat_sum=0;

	for(int i=0;i<mat_rows;i++)
	{
		input_mat_ptr=input_mat.ptr<float>(i);
		output_mat_ptr=output_mat.ptr<int>(i);

		for(int j=0;j<mat_cols;j++)
		{
			input_mat_sum+=abs(input_mat_ptr[j]);
			output_mat_sum+=abs(output_mat_ptr[j]);
		}
	}

	denominator=(output_mat_sum/input_mat_sum+0.5);	//��������
}

//ʵ�������������ֽ�
void computeRationalVector(float gamma,vector<float>& input_vector,vector<int>& output_vector,int& denominator)
{
	//ͳ����������ͳһ��������
	int vec_length=input_vector.size();

	vector<int> output_vector_r(vec_length,0);

	//����Ԫ�ؾ���
	float aver_square=0;
	for(int i=0;i<vec_length;i++)
	{
		aver_square+=input_vector[i]*input_vector[i];
	}
	aver_square=sqrt(aver_square/vec_length);
	float thresold=gamma*aver_square;

	//������С����
	float remain_num=10000;	//��һ���������

	for(int i=0;i<vec_length;i++)
	{
		if(abs(input_vector[i])>0.01*aver_square&&abs(input_vector[i])<remain_num)
			remain_num=abs(input_vector[i]);
	}
	if(remain_num>1)
		remain_num=1;

	//������ʼֵ
	output_vector=vector<int>(vec_length);
	for(int i=0;i<vec_length;i++)
	{
		if(input_vector[i]>=0)
			output_vector[i]=input_vector[i]/remain_num+0.5;
		else
			output_vector[i]=input_vector[i]/remain_num-0.5;
	}

	//ѭ������
	int max_out_num=0;//������

	for(int iterate=0;iterate<10;iterate++)
	{
		float min_num=remain_num;	//���ڼ������С��
		remain_num=0;

		//�����ĸ
		denominator=1/min_num+0.5;

		//������Ӿ���
		for(int i=0;i<vec_length;i++)
		{
			if(input_vector[i]>=0)
				output_vector_r[i]=input_vector[i]/min_num+0.5;
			else
				output_vector_r[i]=input_vector[i]/min_num-0.5;

			if(abs(output_vector_r[i])>max_out_num)
				max_out_num=abs(output_vector_r[i]);

			float remainder=abs(float(output_vector_r[i])/denominator-input_vector[i]);//����
			if(remainder>remain_num)
				remain_num=remainder;
		}

		if(max_out_num>255)//������󲻳���255
		{
			break;
		}

		output_vector=output_vector_r;

		if(remain_num<thresold)//������Сֵ��С����ֵ�����������
			break;
	}

	//�����ĸ
	float input_vec_sum=0;
	int input_vec_sign=0;
	int output_vec_sum=0;

	for(int i=0;i<vec_length;i++)
	{
		input_vec_sum+=abs(input_vector[i]);
		output_vec_sum+=abs(output_vector[i]);

		if(input_vector[i]>0)
			input_vec_sign++;
		if(input_vector[i]<0)
			input_vec_sign--;
	}

	denominator=(output_vec_sum/input_vec_sum+0.5);	//��������
}

//ʵ������������շת�����
void realNumRational(float& gamma,float& input_number,int& output_numerator,int& output_denominator)
{
	if(input_number==0)
	{
		output_numerator=0;
		output_denominator=1;
		return;
	}

	int numerator=1;		//����
	int	denominator;		//��ĸ
	float const_threshold=abs(gamma*input_number);//ֹͣ��ֵ

	for(int i=0;i<10;i++)
	{
		if(input_number>=0)
			denominator=numerator/input_number+0.5;	//��������
		else
			denominator=numerator/input_number-0.5;

		if(i==0)
		{
			output_numerator=numerator;
			output_denominator=denominator;
		}

		float remains=abs(float(numerator)/denominator-input_number);	//��������

		if(abs(remains)<const_threshold||numerator>255)
		{
			break;
		}

		output_numerator=numerator;		//�������
		output_denominator=denominator;	//�����ĸ

		numerator=numerator/remains;
	}
}