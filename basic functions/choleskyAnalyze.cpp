/***************************************************************
	>���ƣ������Cholesky�ֽ�
	>���ߣ���ά��
	>��ϵ��ʽ��williamli_pro@163.com
	>ʵ�ֶ������Գƾ����Cholesky�ֽ�
	>����Ҫ�㣺
	>1.��ͨ�����Cholesky�ֽ�
	>2.ϡ������Cholesky�ֽ�

****************************************************************/

#include "choleskyAnalyze.h"


/*	��ͨ�����Cholesky�ֽ�	
	���룺
	cv::Mat& symmetry_matrix		�����Գƾ���
	�����
	cv::Mat& down_triangle_matrix	�����Ǿ���
	����������falseʱ���������*/
bool normalCholeskyAnalyze(cv::Mat& symmetry_matrix,cv::Mat& down_triangle_matrix)
{
	//1.��ȡ�����ģ
	int symmetry_matrix_rows=symmetry_matrix.rows;
	int symmetry_matrix_cols=symmetry_matrix.cols;

	if(symmetry_matrix_rows!=symmetry_matrix_cols)//������ߴ�ĺ�����
	{
		cout<<"error: -> in normalCholeskyAnalyze : �����������Ӧ����ͬ"<<endl;
		return false;
	}

	//2.����ֽ���
	symmetry_matrix.convertTo(symmetry_matrix,CV_32F);
	down_triangle_matrix=cv::Mat::zeros(symmetry_matrix_rows,symmetry_matrix_cols,symmetry_matrix.type());

	float* symmetry_matrix_ptr;		//	��������ָ��
	float* down_triangle_matrix_ptr_i,*down_triangle_matrix_ptr_j;//	��������ָ��

	for(int i=0;i<symmetry_matrix_rows;i++)
	{
		symmetry_matrix_ptr=symmetry_matrix.ptr<float>(i);
		down_triangle_matrix_ptr_i=down_triangle_matrix.ptr<float>(i);

		for(int j=0;j<=i;j++)
		{
			down_triangle_matrix_ptr_j=down_triangle_matrix.ptr<float>(j);

			float current_value=symmetry_matrix_ptr[j];
			for(int k=0;k<j;k++)
			{
				current_value=current_value-down_triangle_matrix_ptr_i[k]*down_triangle_matrix_ptr_j[k];
			}

			if(i==j)
			{
				if(current_value<0)
				{
					if(current_value>-0.00001)
					{
						current_value=0;
					}
					else
					{
						cout<<"error: -> in normalCholeskyAnalyze : ������������"<<endl;
						return false;
					}
				}
				down_triangle_matrix_ptr_i[j]=sqrt(current_value);
			}
			else
			{
				down_triangle_matrix_ptr_i[j]=current_value/down_triangle_matrix_ptr_j[j];
			}
		}
	}

	return true;
}

/*	ϡ������Cholesky�ֽ�	
	���룺
	vector<Point_Value>& sparse_matrix			�����Գƾ���
	�����
	vector<Point_Value>& down_triangle_matrix	�����Ǿ���
	����������falseʱ���������*/
bool sparseCholeskyAnalyze(vector<Point_Value>& sparse_matrix,vector<Point_Value>& down_triangle_matrix)
{
	//1.��ȡ�����ģ����������Ԫ��
	int sparse_matrix_point_number=sparse_matrix.size();//ϡ������0Ԫ�ظ���

	//ͳ�Ʒ�0Ԫ�ص����з�Χ
	int sparse_matrix_rows=0;	//ϡ���������
	int sparse_matrix_cols=0;	//ϡ���������

	for(int i=0;i<sparse_matrix_point_number;i++)
	{
		if(sparse_matrix[i].row>=sparse_matrix_rows)
			sparse_matrix_rows=sparse_matrix[i].row+1;

		if(sparse_matrix[i].col>=sparse_matrix_cols)
			sparse_matrix_cols=sparse_matrix[i].col+1;
	}

	int sparse_matrix_size=max(sparse_matrix_rows,sparse_matrix_cols);//���������ֵ��Ϊϡ������ģ

	//ͳ�ƾ������Ԫ��
	vector<vector<Point_Value>> value_in_each_line(sparse_matrix_size);		//�洢����Ԫ��
	vector<int> point_number_each_line(sparse_matrix_size,0);				//�洢����Ԫ�ظ���

	int max_number_each_line=5*sparse_matrix_point_number/sparse_matrix_size;	//ÿ��Ԫ��������

	for(int i=0;i<sparse_matrix_size;i++)
	{
		value_in_each_line[i].reserve(max_number_each_line);
	}

	for(int i=0;i<sparse_matrix_point_number;i++)
	{
		int line_id=sparse_matrix[i].row;

		value_in_each_line[line_id].push_back(sparse_matrix[i]);
		point_number_each_line[line_id]++;
	}

	//2.����ֽ���
	vector<vector<Point_Value>> down_triangle_each_line(sparse_matrix_size);	//�洢����������Ԫ��
	vector<int> point_number_triangle_line(sparse_matrix_size,0);				//�洢����������Ԫ�ظ���

	vector<float> cross_value(sparse_matrix_size,0);			//�������Խ�Ԫ��

	for(int i=0;i<sparse_matrix_size;i++)//����洢�ռ�
	{
		down_triangle_each_line[i].reserve(max_number_each_line);
	}

	for(int i=0;i<sparse_matrix_size;i++)
	{
		int j_id=0;//�ж�Ӧ������
		for(int j=0;j<=i;j++)
		{
			float current_value=0;	//��ǰL(i,j)ֵ
			if(point_number_triangle_line[i]==0)	//ĳһ�е��׸�ֵ
			{
				j=value_in_each_line[i][j_id].col;//�к�������һ����0 A(i,j)
				if(j>i)	//��������������㷶Χ
				{
					break;
				}

				current_value=value_in_each_line[i][j_id].value;//A(i,j)

				j_id++;
			}
			else	//��ĳһ�е���ֵ
			{
				if(j>i)	//��������������㷶Χ
				{
					break;
				}

				if(j==value_in_each_line[i][j_id].col)	//����к�j������j_idָ���к���ͬ����A(i,j)!=0
				{
					current_value=value_in_each_line[i][j_id].value;//A(i,j)
					j_id++;
				}

				int angle_line_i_id=0;	//ָ��i�����±�
				int angle_line_j_id=0;	//ָ��j�����±�
				for(int k=0;k<2*j;k++)//current_value=A(i,j)-sum(L(i,k)*L(j,k))
				{
					if(angle_line_i_id>=point_number_triangle_line[i]||angle_line_j_id>=point_number_triangle_line[j])//���������з�Χ������)
					{
						break;
					}

					int line_i_col=down_triangle_each_line[i][angle_line_i_id].col;	//��i��ָ�����
					int line_j_col=down_triangle_each_line[j][angle_line_j_id].col;	//��j��ָ�����

					if(line_i_col==line_j_col)	//��������ȣ���ӦԪ�ز������
					{
						current_value=current_value-down_triangle_each_line[i][angle_line_i_id].value*down_triangle_each_line[j][angle_line_j_id].value;
						angle_line_i_id++;
						angle_line_j_id++;
					}
					else						//�����겻��ȣ��ƶ�����һ��
					{
						if(line_i_col<line_j_col)
						{
							angle_line_i_id++;
						}
						else
						{
							angle_line_j_id++;
						}
					}
				}
			}

			Point_Value current_point;
			current_point.row=i;
			current_point.col=j;
			if(i==j)
			{
				if(current_value<0)
				{
					if(current_value>-0.00001)
					{
						current_value=0;
					}
					else
					{
						cout<<"error: -> in normalCholeskyAnalyze : ������������"<<endl;
						return false;
					}
				}

				current_point.value=sqrt(current_value);//���㵱ǰԪ��ֵ
				cross_value[i]=current_point.value;		//����Խ�Ԫ��
			}
			else
			{
				current_point.value=current_value/cross_value[j];
			}

			//���������󣬸��¸�������
			down_triangle_each_line[i].push_back(current_point);
			point_number_triangle_line[i]++;
		}
	}

	//3.�������Ǿ�������Ϊ������ʽ
	//ͳ�������Ǿ����0Ԫ�ظ���
	int down_triangle_total_element=0;
	for(int i=0;i<sparse_matrix_size;i++)
	{
		down_triangle_total_element+=point_number_triangle_line[i];
	}

	down_triangle_matrix.resize(down_triangle_total_element);

	//������
	int element_id=0;
	for(int i=0;i<sparse_matrix_size;i++)
	{
		for(int j=0;j<point_number_triangle_line[i];j++)
		{
			down_triangle_matrix[element_id]=down_triangle_each_line[i][j];
			element_id++;
		}
	}

	return true;
}

/*	Cholesky�ֽ���Ժ���	*/
bool choleskyAnalyzeTest()
{
	//1 �ܼ�����Cholesky�ֽ�
	cv::Mat symmetry_matrix=cv::Mat::zeros(5,5,CV_32FC1);

	for(int i=1;i<5;i++)
	{
		symmetry_matrix.at<float>(i,i)=2;
		symmetry_matrix.at<float>(i,i-1)=-1;
		symmetry_matrix.at<float>(i-1,i)=-1;
	}
	symmetry_matrix.at<float>(0,0)=1;
	symmetry_matrix.at<float>(4,4)=1;
	cout<<"�ܼ�����Cholesky�ֽ����룺"<<endl<<symmetry_matrix<<endl;

	cv::Mat down_triangle_matrix;
	bool normal_result=normalCholeskyAnalyze(symmetry_matrix,down_triangle_matrix);
	cout<<"�ܼ�����Cholesky�ֽ�����"<<endl<<down_triangle_matrix<<endl;

	if(normal_result==false)
		return false;

	//2 ϡ�����Cholesky�ֽ�
	cv::Matx44f current_matrix(2,-1,0,-1,-1,2,-1,0,0,-1,2,-1,-1,0,-1,2);
	vector<Point_Value> sparse_matrix(24);
	Point_Value point_now;

	int sparse_matrix_id=0;
	for(int i=0;i<4;i++)
	{
		for(int j=0;j<4;j++)
		{
			if(current_matrix(i,j)!=0)
			{
				point_now.row=i;
				point_now.col=j;
				point_now.value=current_matrix(i,j);
				sparse_matrix[sparse_matrix_id]=point_now;
				sparse_matrix_id++;
			}
		}
	}
	for(int i=0;i<12;i++)
	{
		point_now.row=i+4;
		point_now.col=i+4;
		point_now.value=1;
		sparse_matrix[i+12]=point_now;
	}

	vector<Point_Value> down_triangle_matrix_2;
	bool sparse_result=sparseCholeskyAnalyze(sparse_matrix,down_triangle_matrix_2);
	cout<<"ϡ�����Cholesky�ֽ�����"<<endl;

	for(int i=0;i<down_triangle_matrix_2.size();i++)
	{
		cout<<down_triangle_matrix_2[i].row<<" "<<down_triangle_matrix_2[i].col<<" "<<down_triangle_matrix_2[i].value<<endl;
	}

	if(sparse_result==false)
		return false;

	cout<<"����Cholesky�ֽ���Գɹ���"<<endl;
	return true;
}