/***************************************************************
	>名称：矩阵的Cholesky分解
	>作者：李维鹏
	>联系方式：williamli_pro@163.com
	>实现对正定对称矩阵的Cholesky分解
	>技术要点：
	>1.普通矩阵的Cholesky分解
	>2.稀疏矩阵的Cholesky分解

****************************************************************/

#include "choleskyAnalyze.h"


/*	普通矩阵的Cholesky分解	
	输入：
	cv::Mat& symmetry_matrix		正定对称矩阵
	输出：
	cv::Mat& down_triangle_matrix	下三角矩阵
	当函数返回false时，程序出错*/
bool normalCholeskyAnalyze(cv::Mat& symmetry_matrix,cv::Mat& down_triangle_matrix)
{
	//1.获取矩阵规模
	int symmetry_matrix_rows=symmetry_matrix.rows;
	int symmetry_matrix_cols=symmetry_matrix.cols;

	if(symmetry_matrix_rows!=symmetry_matrix_cols)//检测矩阵尺寸的合理性
	{
		cout<<"error: -> in normalCholeskyAnalyze : 输入矩阵行列应该相同"<<endl;
		return false;
	}

	//2.计算分解结果
	symmetry_matrix.convertTo(symmetry_matrix,CV_32F);
	down_triangle_matrix=cv::Mat::zeros(symmetry_matrix_rows,symmetry_matrix_cols,symmetry_matrix.type());

	float* symmetry_matrix_ptr;		//	输入矩阵的指针
	float* down_triangle_matrix_ptr_i,*down_triangle_matrix_ptr_j;//	输出矩阵的指针

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
						cout<<"error: -> in normalCholeskyAnalyze : 输入矩阵非正定"<<endl;
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

/*	稀疏矩阵的Cholesky分解	
	输入：
	vector<Point_Value>& sparse_matrix			正定对称矩阵
	输出：
	vector<Point_Value>& down_triangle_matrix	下三角矩阵
	当函数返回false时，程序出错*/
bool sparseCholeskyAnalyze(vector<Point_Value>& sparse_matrix,vector<Point_Value>& down_triangle_matrix)
{
	//1.获取矩阵规模，分析各行元素
	int sparse_matrix_point_number=sparse_matrix.size();//稀疏矩阵非0元素个数

	//统计非0元素的行列范围
	int sparse_matrix_rows=0;	//稀疏矩阵行数
	int sparse_matrix_cols=0;	//稀疏矩阵列数

	for(int i=0;i<sparse_matrix_point_number;i++)
	{
		if(sparse_matrix[i].row>=sparse_matrix_rows)
			sparse_matrix_rows=sparse_matrix[i].row+1;

		if(sparse_matrix[i].col>=sparse_matrix_cols)
			sparse_matrix_cols=sparse_matrix[i].col+1;
	}

	int sparse_matrix_size=max(sparse_matrix_rows,sparse_matrix_cols);//将行列最大值作为稀疏矩阵规模

	//统计矩阵各行元素
	vector<vector<Point_Value>> value_in_each_line(sparse_matrix_size);		//存储各行元素
	vector<int> point_number_each_line(sparse_matrix_size,0);				//存储各行元素个数

	int max_number_each_line=5*sparse_matrix_point_number/sparse_matrix_size;	//每行元素最大个数

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

	//2.计算分解结果
	vector<vector<Point_Value>> down_triangle_each_line(sparse_matrix_size);	//存储输出矩阵各行元素
	vector<int> point_number_triangle_line(sparse_matrix_size,0);				//存储输出矩阵各行元素个数

	vector<float> cross_value(sparse_matrix_size,0);			//输出矩阵对角元素

	for(int i=0;i<sparse_matrix_size;i++)//分配存储空间
	{
		down_triangle_each_line[i].reserve(max_number_each_line);
	}

	for(int i=0;i<sparse_matrix_size;i++)
	{
		int j_id=0;//列对应索引号
		for(int j=0;j<=i;j++)
		{
			float current_value=0;	//当前L(i,j)值
			if(point_number_triangle_line[i]==0)	//某一行的首个值
			{
				j=value_in_each_line[i][j_id].col;//行号跳到第一个非0 A(i,j)
				if(j>i)	//超过下三角阵计算范围
				{
					break;
				}

				current_value=value_in_each_line[i][j_id].value;//A(i,j)

				j_id++;
			}
			else	//非某一行的首值
			{
				if(j>i)	//超过下三角阵计算范围
				{
					break;
				}

				if(j==value_in_each_line[i][j_id].col)	//如果列号j碰巧与j_id指向列号相同，则A(i,j)!=0
				{
					current_value=value_in_each_line[i][j_id].value;//A(i,j)
					j_id++;
				}

				int angle_line_i_id=0;	//指向i行列下标
				int angle_line_j_id=0;	//指向j行列下标
				for(int k=0;k<2*j;k++)//current_value=A(i,j)-sum(L(i,k)*L(j,k))
				{
					if(angle_line_i_id>=point_number_triangle_line[i]||angle_line_j_id>=point_number_triangle_line[j])//索引超出行范围，跳出)
					{
						break;
					}

					int line_i_col=down_triangle_each_line[i][angle_line_i_id].col;	//第i行指向的列
					int line_j_col=down_triangle_each_line[j][angle_line_j_id].col;	//第j行指向的列

					if(line_i_col==line_j_col)	//列坐标相等，相应元素参与计算
					{
						current_value=current_value-down_triangle_each_line[i][angle_line_i_id].value*down_triangle_each_line[j][angle_line_j_id].value;
						angle_line_i_id++;
						angle_line_j_id++;
					}
					else						//列坐标不相等，移动到下一个
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
						cout<<"error: -> in normalCholeskyAnalyze : 输入矩阵非正定"<<endl;
						return false;
					}
				}

				current_point.value=sqrt(current_value);//计算当前元素值
				cross_value[i]=current_point.value;		//保存对角元素
			}
			else
			{
				current_point.value=current_value/cross_value[j];
			}

			//结果存入矩阵，更新各个参数
			down_triangle_each_line[i].push_back(current_point);
			point_number_triangle_line[i]++;
		}
	}

	//3.将下三角矩阵整理为向量形式
	//统计下三角矩阵非0元素个数
	int down_triangle_total_element=0;
	for(int i=0;i<sparse_matrix_size;i++)
	{
		down_triangle_total_element+=point_number_triangle_line[i];
	}

	down_triangle_matrix.resize(down_triangle_total_element);

	//输出结果
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

/*	Cholesky分解测试函数	*/
bool choleskyAnalyzeTest()
{
	//1 密集矩阵Cholesky分解
	cv::Mat symmetry_matrix=cv::Mat::zeros(5,5,CV_32FC1);

	for(int i=1;i<5;i++)
	{
		symmetry_matrix.at<float>(i,i)=2;
		symmetry_matrix.at<float>(i,i-1)=-1;
		symmetry_matrix.at<float>(i-1,i)=-1;
	}
	symmetry_matrix.at<float>(0,0)=1;
	symmetry_matrix.at<float>(4,4)=1;
	cout<<"密集矩阵Cholesky分解输入："<<endl<<symmetry_matrix<<endl;

	cv::Mat down_triangle_matrix;
	bool normal_result=normalCholeskyAnalyze(symmetry_matrix,down_triangle_matrix);
	cout<<"密集矩阵Cholesky分解结果："<<endl<<down_triangle_matrix<<endl;

	if(normal_result==false)
		return false;

	//2 稀疏矩阵Cholesky分解
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
	cout<<"稀疏矩阵Cholesky分解结果："<<endl;

	for(int i=0;i<down_triangle_matrix_2.size();i++)
	{
		cout<<down_triangle_matrix_2[i].row<<" "<<down_triangle_matrix_2[i].col<<" "<<down_triangle_matrix_2[i].value<<endl;
	}

	if(sparse_result==false)
		return false;

	cout<<"本次Cholesky分解测试成功！"<<endl;
	return true;
}