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

#include "filterAnalysis.h"

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
bool filterMaskAnalysis(cv::Mat& filter_mask,float& gamma,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,vector<Point_Value>& xy_delta_mask,float& standard_error,bool& flag_out)
{
	//1.获取filter_mask尺寸,并定义输出值大小
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	vector<float> vector_mask_y(filter_mask_rows,0);	//y方向（横向）向量模板
	vector<float> vector_mask_x(filter_mask_cols,0);	//x方向（横向）向量模板


	//2.模板x,y向分解
	vector<Point_Value> xy_analyse_delta_mask;	//x,y向分解余量
	float xy_standard_error;					//x,y向分解均方误差
	bool xy_flag_out;							//模板是否可以x,y向分解
	bool complex_reult=complexFilterMaskXYAnalysis(filter_mask,gamma,vector_mask_x,vector_mask_y,xy_analyse_delta_mask,xy_standard_error,xy_flag_out);
	
	filter_mask_x.basic_vector=vector_mask_x;	//x方向基本分解
	filter_mask_y.basic_vector=vector_mask_y;	//y方向基本分解
	filter_mask_x.size=vector_mask_x.size();
	filter_mask_y.size=vector_mask_y.size();
	filter_mask_x.state=0;
	filter_mask_y.state=0;
	xy_delta_mask=xy_analyse_delta_mask;		//输出x,y向分解余量

	if(complex_reult==false)
	{
		return false	;
	}
	if(xy_flag_out==false)
	{
		standard_error=xy_standard_error;
		flag_out=false;
		return true;
	}

	flag_out=true;	//至此，模板至少是可分解的


	//3.进行x,y向模板的等差分解
	//x向模板
	float x_filter_constant,x_filter_linear;	//模板的常数项，一次项
	vector<cv::Vec2f> x_grade_delta_mask;		//稀疏分解余量
	float x_grade_standard_error;				//模板分解的均方标准差
	bool x_grade_flag_out;						//模板是否可拆解

	bool grade_result_x=gradeVectorFilterMaskAnalysis(vector_mask_x,gamma,x_filter_constant,
		x_filter_linear,x_grade_delta_mask,x_grade_standard_error,x_grade_flag_out);

	if(x_grade_flag_out==true)
	{
		filter_mask_x.filter_constant=x_filter_constant;
		filter_mask_x.filter_linear=x_filter_linear;
		filter_mask_x.delta_mask=x_grade_delta_mask;
		filter_mask_x.standard_error=x_grade_standard_error;
		filter_mask_x.state=2;
	}

	//y向模板
	float y_filter_constant,y_filter_linear;	//模板的常数项，一次项
	vector<cv::Vec2f> y_grade_delta_mask;		//稀疏分解余量
	float y_grade_standard_error;				//模板分解的均方标准差
	bool y_grade_flag_out;						//模板是否可拆解

	bool grade_result_y=gradeVectorFilterMaskAnalysis(vector_mask_y,gamma,y_filter_constant,
		y_filter_linear,y_grade_delta_mask,y_grade_standard_error,y_grade_flag_out);

	if(y_grade_flag_out==true)
	{
		filter_mask_y.filter_constant=y_filter_constant;
		filter_mask_y.filter_linear=y_filter_linear;
		filter_mask_y.delta_mask=y_grade_delta_mask;
		filter_mask_y.standard_error=y_grade_standard_error;
		filter_mask_y.state=2;
	}


	//4.进行x,y向模板的二次分解
	//x向模板
	float x_filter_p2;	//模板的常数项，一次项
	float x_p2_standard_error;				//模板分解的均方标准差
	bool x_p2_flag_out;						//模板是否可拆解
	vector<cv::Vec2f> x_p2_delta_mask;		//稀疏分解余量

	if(x_grade_flag_out==false)
	{
		bool p2_result_x=parabolaVectorFilterMaskAnalysis(vector_mask_x,gamma,x_filter_constant,
			x_filter_linear,x_filter_p2,x_p2_delta_mask,x_p2_standard_error,x_p2_flag_out);

		if(x_p2_flag_out==true)
		{
			filter_mask_x.filter_constant=x_filter_constant;
			filter_mask_x.filter_linear=x_filter_linear;
			filter_mask_x.filter_p2=x_filter_p2;
			filter_mask_x.delta_mask=x_p2_delta_mask;
			filter_mask_x.standard_error=x_p2_standard_error;
			filter_mask_x.state=3;

			if(x_filter_p2==0)
				filter_mask_x.state=2;
		}
	}

	//y向模板
	float y_filter_p2;	//模板的常数项，一次项
	float y_p2_standard_error;				//模板分解的均方标准差
	bool y_p2_flag_out;						//模板是否可拆解
	vector<cv::Vec2f> y_p2_delta_mask;		//稀疏分解余量

	if(y_grade_flag_out==false)
	{
		bool p2_result_y=parabolaVectorFilterMaskAnalysis(vector_mask_y,gamma,y_filter_constant,
			y_filter_linear,y_filter_p2,y_p2_delta_mask,y_p2_standard_error,y_p2_flag_out);

		if(y_p2_flag_out==true)
		{
			filter_mask_y.filter_constant=y_filter_constant;
			filter_mask_y.filter_linear=y_filter_linear;
			filter_mask_y.filter_p2=y_filter_p2;
			filter_mask_y.delta_mask=y_p2_delta_mask;
			filter_mask_y.standard_error=y_p2_standard_error;
			filter_mask_y.state=3;

			if(y_filter_p2==0)
			filter_mask_y.state=2;
		}
	}


	//5.进行x,y向模板的对称分解
	//x向模板
	bool symmetry_result_x=false;
	vector<float> x_symmetry_mask;				//对称化向量
	vector<cv::Vec2f> x_symmetry_delta_mask;	//稀疏分解余量
	float x_symmetry_standard_error;			//模板分解的均方标准差
	bool x_symmetry_flag_out;					//模板是否可拆解

	if(x_grade_flag_out==false&&x_p2_flag_out==false)		//仅当等差化与二次化失败时进行对称化处理
	{
		symmetry_result_x=vectorFilterMaskSymmetryAnalysis(vector_mask_x,gamma,x_symmetry_mask,
		x_symmetry_delta_mask,x_symmetry_standard_error,x_symmetry_flag_out);

		if(x_symmetry_flag_out)
		{
			filter_mask_x.symmetry_vector=x_symmetry_mask;
			filter_mask_x.delta_mask=x_symmetry_delta_mask;
			filter_mask_x.standard_error=x_symmetry_standard_error;
			filter_mask_x.state=1;
		}
	}

	//y向模板
	bool symmetry_result_y=false;
	vector<float> y_symmetry_mask;				//对称化向量
	vector<cv::Vec2f> y_symmetry_delta_mask;	//稀疏分解余量
	float y_symmetry_standard_error;			//模板分解的均方标准差
	bool y_symmetry_flag_out;					//模板是否可拆解

	if(y_grade_flag_out==false&&y_p2_flag_out==false)		//仅当等差化二次化失败时进行对称化处理
	{
		symmetry_result_y=vectorFilterMaskSymmetryAnalysis(vector_mask_y,gamma,y_symmetry_mask,
		y_symmetry_delta_mask,y_symmetry_standard_error,y_symmetry_flag_out);

		if(y_symmetry_flag_out)
		{
			filter_mask_y.symmetry_vector=y_symmetry_mask;
			filter_mask_y.delta_mask=y_symmetry_delta_mask;
			filter_mask_y.standard_error=y_symmetry_standard_error;
			filter_mask_y.state=1;
		}
	}


	//6.分析综合误差，输出模板分解结果
	//分析综合误差  total_s^2=xy_s^2+x_s^2*|Tym|^2/m+y_s^2*|Txm|^2/m+x_s^2*y_s^2;
	standard_error=xy_standard_error*xy_standard_error;

	if(filter_mask_x.state>0)	//当error[Tx]!=0时，计算|Tym|
	{
		float vector_mask_y_square=0;
		for(int i=0;i<filter_mask_rows;i++)
		{
			vector_mask_y_square+=vector_mask_y[i]*vector_mask_y[i];
		}

		standard_error=standard_error+filter_mask_x.standard_error*vector_mask_y_square/filter_mask_rows;
	}

	if(filter_mask_y.state>0)	//当error[Ty]!=0时，计算|Txm|
	{
		float vector_mask_x_square=0;
		for(int i=0;i<filter_mask_cols;i++)
		{
			vector_mask_x_square+=vector_mask_x[i]*vector_mask_x[i];
		}

		standard_error=standard_error+filter_mask_x.standard_error*vector_mask_x_square/filter_mask_cols;
	}
	
	if(filter_mask_x.state>0&&filter_mask_y.state>0)
	{
		standard_error+=filter_mask_x.standard_error*filter_mask_y.standard_error;
	}

	standard_error=sqrt(standard_error);

	return true;
}

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
bool filterMaskAnalysis(cv::Mat& filter_mask,float& gamma,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,
	float& mask_const,vector<Point_Value>& xy_delta_mask,float& standard_error,bool& flag_out)			//浮点数形式向量
{
	//1.获取filter_mask尺寸,并定义输出值大小
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	vector<float> vector_mask_y(filter_mask_rows,0);	//y方向（横向）向量模板
	vector<float> vector_mask_x(filter_mask_cols,0);	//x方向（横向）向量模板


	//2.模板x,y向分解
	vector<Point_Value> xy_analyse_delta_mask;	//x,y向分解余量
	float xy_standard_error;					//x,y向分解均方误差
	bool xy_flag_out;							//模板是否可以x,y向分解

	//(1)模板xy分解
	mask_const=0;	//常量初值为0
	bool complex_reult=complexFilterMaskXYAnalysis(filter_mask,gamma,vector_mask_x,vector_mask_y,xy_analyse_delta_mask,xy_standard_error,xy_flag_out);
	
	filter_mask_x.basic_vector=vector_mask_x;	//x方向基本分解
	filter_mask_y.basic_vector=vector_mask_y;	//y方向基本分解
	filter_mask_x.size=vector_mask_x.size();
	filter_mask_y.size=vector_mask_y.size();
	filter_mask_x.state=0;
	filter_mask_y.state=0;
	xy_delta_mask=xy_analyse_delta_mask;		//输出x,y向分解余量

	if(xy_flag_out==false)	//模板xy分解失败，执行xy,const分解
	{
		//(2)模板xy,const分解
		complex_reult=complexFilterMaskXYconstAnalysis(filter_mask,gamma,vector_mask_x,vector_mask_y,mask_const,xy_analyse_delta_mask,xy_standard_error,xy_flag_out);

		filter_mask_x.basic_vector=vector_mask_x;	//x方向基本分解
		filter_mask_y.basic_vector=vector_mask_y;	//y方向基本分解
		filter_mask_x.size=vector_mask_x.size();
		filter_mask_y.size=vector_mask_y.size();
		filter_mask_x.state=0;
		filter_mask_y.state=0;
		xy_delta_mask=xy_analyse_delta_mask;		//输出x,y向分解余量

		if(complex_reult==false)
		{
			return false;
		}
		if(xy_flag_out==false)
		{
			standard_error=xy_standard_error;
			flag_out=false;
			return true;
		}
	}

	flag_out=true;	//至此，模板至少是可分解的


	//3.进行x,y向模板的等差分解
	//判断x,y向量是否有效
	if(filter_mask_x.basic_vector.size()==0||filter_mask_y.basic_vector.size()==0)	//分解结果只有常数项与分离项
	{
		return true;
	}

	//x向模板
	float x_filter_constant,x_filter_linear;	//模板的常数项，一次项
	vector<cv::Vec2f> x_grade_delta_mask;		//稀疏分解余量
	float x_grade_standard_error;				//模板分解的均方标准差
	bool x_grade_flag_out;						//模板是否可拆解

	bool grade_result_x=gradeVectorFilterMaskAnalysis(vector_mask_x,gamma,x_filter_constant,
		x_filter_linear,x_grade_delta_mask,x_grade_standard_error,x_grade_flag_out);

	if(x_grade_flag_out==true)
	{
		filter_mask_x.filter_constant=x_filter_constant;
		filter_mask_x.filter_linear=x_filter_linear;
		filter_mask_x.delta_mask=x_grade_delta_mask;
		filter_mask_x.standard_error=x_grade_standard_error;
		filter_mask_x.state=2;
	}

	//y向模板
	float y_filter_constant,y_filter_linear;	//模板的常数项，一次项
	vector<cv::Vec2f> y_grade_delta_mask;		//稀疏分解余量
	float y_grade_standard_error;				//模板分解的均方标准差
	bool y_grade_flag_out;						//模板是否可拆解

	bool grade_result_y=gradeVectorFilterMaskAnalysis(vector_mask_y,gamma,y_filter_constant,
		y_filter_linear,y_grade_delta_mask,y_grade_standard_error,y_grade_flag_out);

	if(y_grade_flag_out==true)
	{
		filter_mask_y.filter_constant=y_filter_constant;
		filter_mask_y.filter_linear=y_filter_linear;
		filter_mask_y.delta_mask=y_grade_delta_mask;
		filter_mask_y.standard_error=y_grade_standard_error;
		filter_mask_y.state=2;
	}


	//4.进行x,y向模板的二次分解
	//x向模板
	float x_filter_p2;	//模板的常数项，一次项
	float x_p2_standard_error;				//模板分解的均方标准差
	bool x_p2_flag_out;						//模板是否可拆解
	vector<cv::Vec2f> x_p2_delta_mask;		//稀疏分解余量

	bool p2_result_x=parabolaVectorFilterMaskAnalysis(vector_mask_x,gamma,x_filter_constant,
		x_filter_linear,x_filter_p2,x_p2_delta_mask,x_p2_standard_error,x_p2_flag_out);

	if(x_p2_flag_out==true)
	{
		filter_mask_x.filter_constant=x_filter_constant;
		filter_mask_x.filter_linear=x_filter_linear;
		filter_mask_x.filter_p2=x_filter_p2;
		filter_mask_x.delta_mask=x_p2_delta_mask;
		filter_mask_x.standard_error=x_p2_standard_error;
		filter_mask_x.state=3;

		if(x_filter_p2==0)
			filter_mask_x.state=2;
	}

	//y向模板
	float y_filter_p2;	//模板的常数项，一次项
	float y_p2_standard_error;				//模板分解的均方标准差
	bool y_p2_flag_out;						//模板是否可拆解
	vector<cv::Vec2f> y_p2_delta_mask;		//稀疏分解余量

	bool p2_result_y=parabolaVectorFilterMaskAnalysis(vector_mask_y,gamma,y_filter_constant,
		y_filter_linear,y_filter_p2,y_p2_delta_mask,y_p2_standard_error,y_p2_flag_out);

	if(y_p2_flag_out==true)
	{
		filter_mask_y.filter_constant=y_filter_constant;
		filter_mask_y.filter_linear=y_filter_linear;
		filter_mask_y.filter_p2=y_filter_p2;
		filter_mask_y.delta_mask=y_p2_delta_mask;
		filter_mask_y.standard_error=y_p2_standard_error;
		filter_mask_y.state=3;

		if(y_filter_p2==0)
			filter_mask_y.state=2;
	}


	//5.进行x,y向模板的对称分解
	//x向模板
	bool symmetry_result_x=false;
	vector<float> x_symmetry_mask;				//对称化向量
	vector<cv::Vec2f> x_symmetry_delta_mask;	//稀疏分解余量
	float x_symmetry_standard_error;			//模板分解的均方标准差
	bool x_symmetry_flag_out;					//模板是否可拆解

	if(x_p2_flag_out==false)					//仅当等差化失败时进行对称化处理
	{
		symmetry_result_x=vectorFilterMaskSymmetryAnalysis(vector_mask_x,gamma,x_symmetry_mask,
		x_symmetry_delta_mask,x_symmetry_standard_error,x_symmetry_flag_out);

		filter_mask_x.symmetry_vector=x_symmetry_mask;
		filter_mask_x.delta_mask=x_symmetry_delta_mask;
		filter_mask_x.standard_error=x_symmetry_standard_error;
		filter_mask_x.state=1;
	}

	//y向模板
	bool symmetry_result_y=false;
	vector<float> y_symmetry_mask;				//对称化向量
	vector<cv::Vec2f> y_symmetry_delta_mask;	//稀疏分解余量
	float y_symmetry_standard_error;			//模板分解的均方标准差
	bool y_symmetry_flag_out;					//模板是否可拆解

	if(y_p2_flag_out==false)					//仅当等差化失败时进行对称化处理
	{
		symmetry_result_y=vectorFilterMaskSymmetryAnalysis(vector_mask_y,gamma,y_symmetry_mask,
		y_symmetry_delta_mask,y_symmetry_standard_error,y_symmetry_flag_out);

		filter_mask_y.symmetry_vector=y_symmetry_mask;
		filter_mask_y.delta_mask=y_symmetry_delta_mask;
		filter_mask_y.standard_error=y_symmetry_standard_error;
		filter_mask_y.state=1;
	}


	//6.分析综合误差，输出模板分解结果
	//分析综合误差  total_s^2=xy_s^2+x_s^2*|Tym|^2/m+y_s^2*|Txm|^2/m+x_s^2*y_s^2;
	standard_error=xy_standard_error*xy_standard_error;

	if(filter_mask_x.state>0)	//当error[Tx]!=0时，计算|Tym|
	{
		float vector_mask_y_square=0;
		for(int i=0;i<filter_mask_rows;i++)
		{
			vector_mask_y_square+=vector_mask_y[i]*vector_mask_y[i];
		}

		standard_error=standard_error+filter_mask_x.standard_error*vector_mask_y_square/filter_mask_rows;
	}

	if(filter_mask_y.state>0)	//当error[Ty]!=0时，计算|Txm|
	{
		float vector_mask_x_square=0;
		for(int i=0;i<filter_mask_cols;i++)
		{
			vector_mask_x_square+=vector_mask_x[i]*vector_mask_x[i];
		}

		standard_error=standard_error+filter_mask_x.standard_error*vector_mask_x_square/filter_mask_cols;
	}
	
	if(filter_mask_x.state>0&&filter_mask_y.state)
	{
		standard_error+=filter_mask_x.standard_error*filter_mask_y.standard_error;
	}

	standard_error=sqrt(standard_error);

	return true;
}


/*	1.1 模板的基本x,y向分解	
	输入：
	cv::Mat& filter_mask				模板矩阵
	输出：
	vector<float>& filter_mask_x		x方向（横向）向量模板
	vector<float>& filter_mask_y		y方向（纵向）向量模板
	float& standard_error				均方标准差
	bool flag_out						输入模板是否可拆解（ture为可拆解，此时模板的x,y分量拆解有效）
	当函数返回false时，程序出错*/
bool basicFilterMaskXYAnalysis(cv::Mat& filter_mask,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& standard_error,bool& flag_out)
{
	//1.获取filter_mask尺寸,并定义输出值大小
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;


	//2.计算filter_mask所有元素的平方和，x,y模板归一化参数
	float* filter_mask_ptr;		//模板迭代指针
	float element_square_sum=0;	//模板元素的平方和

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			element_square_sum+=filter_mask_ptr[j]*filter_mask_ptr[j];
		}
	}
	
	float x_square_sum=0;
	for(int i=0;i<filter_mask_x.size();i++)
		x_square_sum+=filter_mask_x[i]*filter_mask_x[i];
	float y_square_sum=0;
	for(int i=0;i<filter_mask_y.size();i++)
		y_square_sum+=filter_mask_y[i]*filter_mask_y[i];
	float xy_square_sum=x_square_sum*y_square_sum;	//x,y的平方和
		
	float alpha=sqrt(element_square_sum);	//x,y模板归一化参数
	if(xy_square_sum>0.6*element_square_sum)
	{
		alpha=sqrt(xy_square_sum);
	}

	float total_error=0;//总的平方误差 sum[(T(i,j)-Tx(i)Ty(j))^2]

	for(int iterate=0;iterate<2;iterate++)	//由于alpha的初值基于假设standard_error=0，需要迭代校正alpha
	{
		//3.计算y模板所满足方程 (alpha^2*I-T'T)y=0 中矩阵 B=(alpha^2*I-T'T) 的Cholesky分解
		cv::Mat symmetry_matrix=cv::Mat::zeros(filter_mask_rows,filter_mask_rows,CV_32F);	//待分解正定对称矩阵
		for(int i=0;i<filter_mask_rows;i++)
		{
			symmetry_matrix.at<float>(i,i)=element_square_sum;
		}
		symmetry_matrix=symmetry_matrix-filter_mask*filter_mask.t();

		cv::Mat down_triangle_matrix;	//分解结果(下三角阵)

		bool analyse_result=normalCholeskyAnalyze(symmetry_matrix,down_triangle_matrix);//调用Cholesky分解
		if(analyse_result==false)
		{
			cout<<"error: -> in basicFilterMaskXYAnalysis"<<endl;
			return false;
		}


		//4.根据Cholesky分解结果求x,y模板：L'y=0,y'y=alpha
		//初始化想x,y向模板
		filter_mask_y=vector<float>(filter_mask_rows,0);
		filter_mask_x=vector<float>(filter_mask_cols,0);

		//计算y模板
		filter_mask_y[filter_mask_rows-1]=1;	//设置一个参考初值

		cv::Mat down_triangle_matrix_t=down_triangle_matrix.t();
		float* down_triangle_matrix_t_ptr;	//L'矩阵的行指针

		for(int i=filter_mask_rows-2;i>=0;i--)
		{
			down_triangle_matrix_t_ptr=down_triangle_matrix_t.ptr<float>(i);

			// y(i)=-sum(L'(i,j)*y(j))/L'(i,i),j=i+1->filter_mask_rows
			float current_sum_y_value=0;
			for(int j=i+1;j<filter_mask_rows;j++)
			{
				current_sum_y_value+=down_triangle_matrix_t_ptr[j]*filter_mask_y[j];
			}

			filter_mask_y[i]=-current_sum_y_value/down_triangle_matrix_t_ptr[i];
		}


		//y模板归一化
		float current_y_model=0;	//计算y的模长
		for(int i=0;i<filter_mask_rows;i++)
		{
			current_y_model+=filter_mask_y[i]*filter_mask_y[i];
		}
		current_y_model=sqrt(current_y_model);

		float new_y_model=sqrt(alpha);
		for(int i=0;i<filter_mask_rows;i++)	//归一化
		{
			filter_mask_y[i]=filter_mask_y[i]*new_y_model/current_y_model;
		}

		//计算x模板
		for(int i=0;i<filter_mask_rows;i++)
		{
			filter_mask_ptr=filter_mask.ptr<float>(i);

			for(int j=0;j<filter_mask_cols;j++)
			{
				filter_mask_x[j]+=filter_mask_ptr[j]*filter_mask_y[i];
			}
		}

		for(int i=0;i<filter_mask_cols;i++)
		{
			filter_mask_x[i]=filter_mask_x[i]/alpha;
		}

		//4.计算分量模板对各个元素方差之和
		total_error=0;

		for(int i=0;i<filter_mask_rows;i++)
		{
			filter_mask_ptr=filter_mask.ptr<float>(i);

			for(int j=0;j<filter_mask_cols;j++)
			{
				float current_error=filter_mask_y[i]*filter_mask_x[j]-filter_mask_ptr[j];
				total_error+=current_error*current_error;
			}
		}

		//校正alpha的估计值
		alpha=sqrt(element_square_sum-total_error/(1-(filter_mask_rows+filter_mask_cols)/(filter_mask_rows*filter_mask_cols)));
	}
	

	//5.根据元素方差之和判断模板的可分解性
	//计算目标函数
	float aver_standard=total_error/(filter_mask_rows*filter_mask_cols-filter_mask_rows-filter_mask_cols);
	standard_error=sqrt(aver_standard);	//输出均方标准差结果

	//计算阈值，阈值由模板元素平方和决定
	float threshold=0.01*sqrt(element_square_sum/(filter_mask_rows*filter_mask_cols));

	//进行比较
	flag_out=false;
	if(aver_standard<threshold)
	{
		flag_out=true;
	}

	return true;
}


/*	1.2 复杂模板的x,y向分解	
	输入：
	cv::Mat& filter_mask				模板矩阵
	float& gamma						信噪比
	输出：
	vector<float>& filter_mask_x		x方向（横向）向量模板
	vector<float>& filter_mask_y		y方向（纵向）向量模板
	vector<Point_Value>& delta_mask		稀疏误差矩阵(x,y,df)
	float& standard_error				均方标准差
	bool flag_out						输入模板是否可拆解（ture为可拆解，此时模板的x,y分量拆解有效）
	当函数返回false时，程序出错*/
bool complexFilterMaskXYAnalysis(cv::Mat& filter_mask,float& gamma,vector<float>& filter_mask_x,vector<float>& filter_mask_y,vector<Point_Value>& delta_mask,float& standard_error,bool& flag_out)
{
	//1.获取filter_mask尺寸,并定义输出值大小
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	filter_mask_y=vector<float>(filter_mask_rows,0);
	filter_mask_x=vector<float>(filter_mask_cols,0);

	//先尝试一次直接分解
	basicFilterMaskXYAnalysis(filter_mask,filter_mask_x,filter_mask_y,standard_error,flag_out);
	if(flag_out==true)
	{
		standard_error=standard_error;
		delta_mask.clear();
		return true;
	}


	//2.计算所有行、列的向量模长
	vector<float> each_x_vector_model(filter_mask_rows,0);	//所有x向量模长
	vector<float> each_y_vector_model(filter_mask_cols,0);	//所有y向量模长

	vector<int> each_x_vector_sign(filter_mask_rows,0);	//各个x向量的元素符号和
	vector<int> each_y_vector_sign(filter_mask_cols,0);	//各个y向量的元素符号和

	float sum_element_sqaure=0;	//元素平方和

	float* filter_mask_ptr;		//模板迭代指针

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			float square_value=filter_mask_ptr[j]*filter_mask_ptr[j];//元素的平方
			sum_element_sqaure+=square_value;

			each_x_vector_model[i]+=square_value;
			each_y_vector_model[j]+=square_value;

			if(filter_mask_ptr[j]<0)
			{
				each_x_vector_sign[i]--;
				each_y_vector_sign[j]--;
			}
			if(filter_mask_ptr[j]>0)
			{
				each_x_vector_sign[i]++;
				each_y_vector_sign[j]++;
			}
		}
	}

	for(int i=0;i<filter_mask_rows;i++)
	{
		each_x_vector_model[i]=sqrt(each_x_vector_model[i]);
	}
	for(int i=0;i<filter_mask_cols;i++)
	{
		each_y_vector_model[i]=sqrt(each_y_vector_model[i]);
	}


	//3.计算所有行、列的归一化向量
	cv::Mat normalized_x_vectors;	//x方向归一化向量
	cv::Mat normalized_y_vectors;	//y方向归一化向量

	filter_mask.copyTo(normalized_x_vectors);
	filter_mask.copyTo(normalized_y_vectors);

	float* normalized_x_vectors_ptr;		//x方向归一化向量迭代指针
	float* normalized_y_vectors_ptr;		//y方向归一化向量代指针

	for(int i=0;i<filter_mask_rows;i++)
	{
		normalized_x_vectors_ptr=normalized_x_vectors.ptr<float>(i);
		normalized_y_vectors_ptr=normalized_y_vectors.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			normalized_x_vectors_ptr[j]=normalized_x_vectors_ptr[j]/each_x_vector_model[i];
			normalized_y_vectors_ptr[j]=normalized_y_vectors_ptr[j]/each_y_vector_model[j];

			if(each_x_vector_sign[i]<0)//归一化主方向（主方向在第一象限，大于0）
			{
				normalized_x_vectors_ptr[j]=-normalized_x_vectors_ptr[j];
			}
			if(each_y_vector_sign[j]<0)//归一化主方向（主方向在第一象限，大于0）
			{
				normalized_y_vectors_ptr[j]=-normalized_y_vectors_ptr[j];
			}
		}
	}

	//4.根据每行、列的归一化向量的估计结果，计算每行、列各个元素均值以及标准差
	//计算均值
	vector<float> element_sum_in_x_vectors(filter_mask_cols,0);	//x向量中各个元素均值
	vector<float> element_sum_in_y_vectors(filter_mask_rows,0);	//y向量中各个元素均值

	vector<float> element_aver_in_x_vectors(filter_mask_cols,0);	//x向量中各个元素均值
	vector<float> element_aver_in_y_vectors(filter_mask_rows,0);	//y向量中各个元素均值

	for(int i=0;i<filter_mask_rows;i++)
	{
		normalized_x_vectors_ptr=normalized_x_vectors.ptr<float>(i);
		normalized_y_vectors_ptr=normalized_y_vectors.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			element_sum_in_x_vectors[j]+=normalized_x_vectors_ptr[j];
			element_sum_in_y_vectors[i]+=normalized_y_vectors_ptr[j];
		}
	}

	for(int i=0;i<filter_mask_cols;i++)
	{
		element_aver_in_x_vectors[i]=element_sum_in_x_vectors[i]/filter_mask_rows;
	}
	for(int i=0;i<filter_mask_rows;i++)
	{
		element_aver_in_y_vectors[i]=element_sum_in_y_vectors[i]/filter_mask_cols;
	}

	//计算方差
	vector<float> element_standard_in_x_vectors(filter_mask_cols,0);	//x向量中各个元素标准差
	vector<float> element_standard_in_y_vectors(filter_mask_rows,0);	//y向量中各个元素标准差

	cv::Mat normalized_x_vectors_error(filter_mask_rows,filter_mask_cols,CV_32F);	//x方向归一化向量误差
	cv::Mat normalized_y_vectors_error(filter_mask_rows,filter_mask_cols,CV_32F);	//y方向归一化向量误差

	for(int i=0;i<filter_mask_rows;i++)
	{
		normalized_x_vectors_ptr=normalized_x_vectors.ptr<float>(i);
		normalized_y_vectors_ptr=normalized_y_vectors.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			float dx_element=normalized_x_vectors_ptr[j]-element_aver_in_x_vectors[j];	//x元素误差
			float dy_element=normalized_y_vectors_ptr[j]-element_aver_in_y_vectors[i];	//y元素误差

			element_standard_in_x_vectors[j]+=dx_element*dx_element;
			element_standard_in_y_vectors[i]+=dy_element*dy_element;
		}
	}

	for(int i=0;i<filter_mask_rows;i++)
	{
		element_standard_in_y_vectors[i]=sqrt(element_standard_in_y_vectors[i]/filter_mask_rows);
	}
	for(int i=0;i<filter_mask_cols;i++)
	{
		element_standard_in_x_vectors[i]=sqrt(element_standard_in_x_vectors[i]/filter_mask_cols);
	}

	//计算标准差倒数加权均值
	vector<float> element_w_aver_in_x_vectors(filter_mask_cols,0);	//x向量中各个元素加权均值
	vector<float> element_w_aver_in_y_vectors(filter_mask_rows,0);	//y向量中各个元素加权均值

	float weight_x=0;//各行x向量权重
	float weight_y=0;//各列y向量权重

	for(int i=0;i<filter_mask_rows;i++)//计算各个x向量权重
	{
		if(element_standard_in_y_vectors[i]<0.00001)
			weight_x+=100;
		else
			weight_x+=1/element_standard_in_y_vectors[i];
	}

	for(int i=0;i<filter_mask_cols;i++)//计算各个y向量权重
	{
		if(element_standard_in_x_vectors[i]<0.00001)
			weight_y+=100;
		else
			weight_y+=1/element_standard_in_x_vectors[i];
	}

	for(int i=0;i<filter_mask_rows;i++)
	{
		normalized_x_vectors_ptr=normalized_x_vectors.ptr<float>(i);
		normalized_y_vectors_ptr=normalized_y_vectors.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			if(element_standard_in_y_vectors[i]<0.00001)
			{
				element_w_aver_in_x_vectors[j]+=normalized_x_vectors_ptr[j]*100;
			}
			else
			{
				element_w_aver_in_x_vectors[j]+=normalized_x_vectors_ptr[j]/element_standard_in_y_vectors[i];
			}
			
			if(element_standard_in_x_vectors[j]<0.00001)
			{
				element_w_aver_in_y_vectors[i]+=normalized_y_vectors_ptr[j]*100;
			}
			else
			{
				element_w_aver_in_y_vectors[i]+=normalized_y_vectors_ptr[j]/element_standard_in_x_vectors[j];
			}
		}
	}

	for(int i=0;i<filter_mask_cols;i++)
	{
		element_w_aver_in_x_vectors[i]=element_w_aver_in_x_vectors[i]/weight_x;
	}
	for(int i=0;i<filter_mask_rows;i++)
	{
		element_w_aver_in_y_vectors[i]=element_w_aver_in_y_vectors[i]/weight_y;
	}


	//5.记录下与加权均值之差大于2倍标准差的x、y向量元素对应的列、行坐标，并做不重复统计，得到潜在误差分离对象
	
	//分析各行、列是否超差
	vector<bool> whether_over_error_in_x_vectors(filter_mask_rows,false);	//x向量中各行是否超出误差范围
	vector<bool> whether_over_error_in_y_vectors(filter_mask_cols,false);	//y向量中各列是否超出误差范围

	for(int i=0;i<filter_mask_rows;i++)
	{
		normalized_x_vectors_ptr=normalized_x_vectors.ptr<float>(i);
		normalized_y_vectors_ptr=normalized_y_vectors.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			if(abs(normalized_x_vectors_ptr[j]-element_w_aver_in_x_vectors[j])>element_standard_in_x_vectors[j]+gamma/filter_mask_cols)//第i行x向量中第j个元素误差过大
			{
				whether_over_error_in_x_vectors[i]=true;
			}
			if(abs(normalized_y_vectors_ptr[j]-element_w_aver_in_y_vectors[i])>element_standard_in_y_vectors[i]+gamma/filter_mask_rows)//第j列y向量中第i个元素误差过大
			{
				whether_over_error_in_y_vectors[j]=true;
			}
		}
	}

	//统计超差的行、列坐标
	vector<int> over_error_y_in_x_vectors;	//x向量中超出误差范围的行
	vector<int> over_error_x_in_y_vectors;	//y向量中超出误差范围的列

	over_error_y_in_x_vectors.reserve(filter_mask_rows);
	over_error_x_in_y_vectors.reserve(filter_mask_cols);

	int over_error_rows=0;	//超差行数
	int over_error_cols=0;	//超差列数

	for(int i=0;i<filter_mask_rows;i++)
	{
		if(whether_over_error_in_x_vectors[i])
		{
			over_error_y_in_x_vectors.push_back(i);
			over_error_rows++;
		}
	}
	for(int i=0;i<filter_mask_cols;i++)
	{
		if(whether_over_error_in_y_vectors[i])
		{
			over_error_x_in_y_vectors.push_back(i);
			over_error_cols++;
		}
	}

	if(over_error_rows>filter_mask_rows-2||over_error_cols>filter_mask_cols-2)//如果超差行列过多，说明没有分解的必要
	{
		flag_out=false;	//模板无法分解
		return true;
	}


	//6.采用除了潜在误差分离对象所在行/列以外的数据计算y,x模板分量的初步估计值
	//	计算除潜在误差分离对象所在行/列以外各个x,y归一化向量之和
	vector<float> primary_x_vector=element_sum_in_x_vectors;	//x向量初步估计
	vector<float> primary_y_vector=element_sum_in_y_vectors;	//y向量初步估计

	//剔除误差分离对象所在行元素值
	for(int i=0;i<over_error_rows;i++)
	{
		normalized_x_vectors_ptr=normalized_x_vectors.ptr<float>(over_error_y_in_x_vectors[i]);

		for(int j=0;j<filter_mask_cols;j++)
		{
			primary_x_vector[j]-=normalized_x_vectors_ptr[j];
		}
	}
	//剔除误差分离对象所在列元素值
	for(int i=0;i<filter_mask_rows;i++)
	{
		normalized_y_vectors_ptr=normalized_y_vectors.ptr<float>(i);

		for(int j=0;j<over_error_cols;j++)
		{
			primary_y_vector[i]-=normalized_y_vectors_ptr[over_error_x_in_y_vectors[j]];
		}
	}

	//	得到最小均方误差的比例系数min J=sum[(T(i,j)-alpha^2*Vx(i)*Vy(j))^2]
	float alpha_x=0;	//最小均方误差的比例系数
	float alpha_y=0;

	float primary_x_vectors_part_model_square=0;	//x向量初步估计的部分模的平方(不包括误差分离对象)
	float primary_y_vectors_part_model_square=0;	//y向量初步估计的部分模的平方

	for(int i=0;i<filter_mask_cols;i++)//计算x向量模
	{
		if(whether_over_error_in_y_vectors[i]==false)//非误差超差列
		{
			primary_x_vectors_part_model_square+=primary_x_vector[i]*primary_x_vector[i];
		}
	}

	for(int i=0;i<filter_mask_rows;i++)//计算y向量模
	{
		if(whether_over_error_in_x_vectors[i]==false)//非误差超差行
		{
			primary_y_vectors_part_model_square+=primary_y_vector[i]*primary_y_vector[i];
		}
	}

	float x_y_vector_element_2_sum=primary_x_vectors_part_model_square*primary_y_vectors_part_model_square;	//sum(primary_x_vector^2*primary_y_vector^2)x,y向量之积元素的平方和

	for(int i=0;i<filter_mask_rows;i++)
	{
		if(whether_over_error_in_x_vectors[i])//跳过误差超差行
		{
			continue;
		}
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			if(whether_over_error_in_y_vectors[j])//跳过误差超差列
			{
				continue;
			}

			alpha_x+=filter_mask_ptr[j]*primary_y_vector[i]*primary_x_vector[j];//计算最小均方误差的比例系数
		}
	}
	if(alpha_x>0)
	{
		alpha_x=sqrt(alpha_x/x_y_vector_element_2_sum);
		alpha_y=alpha_x;
	}
	else
	{
		alpha_x=-sqrt(-alpha_x/x_y_vector_element_2_sum);
		alpha_y=-alpha_x;
	}
	

	//	得到x,y向模板分量初步估计值
	for(int i=0;i<filter_mask_cols;i++)//计算x向量模
	{
		primary_x_vector[i]=primary_x_vector[i]*alpha_x;
	}

	for(int i=0;i<filter_mask_rows;i++)//计算y向量模
	{
		primary_y_vector[i]=primary_y_vector[i]*alpha_y;
	}

	//	计算模板元素标准差的初步估计值
	standard_error=0;	//标准差

	for(int i=0;i<filter_mask_rows;i++)
	{
		if(whether_over_error_in_x_vectors[i])//跳过误差超差行
		{
			continue;
		}
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			if(whether_over_error_in_y_vectors[j])//跳过误差超差列
			{
				continue;
			}

			float current_error=filter_mask_ptr[j]-primary_y_vector[i]*primary_x_vector[j];//误差
			standard_error+=current_error*current_error;//方差
		}
	}
	standard_error=sqrt(standard_error/(filter_mask_rows-over_error_rows)/(filter_mask_cols-over_error_cols));


	//7.迭代估计x,y模板，并根据标准差从潜在分离对象中筛选出真实分离对象

	//计算超差判别参数
	float over_error_threshold=gamma*sqrt(sum_element_sqaure/(filter_mask_rows*filter_mask_cols));//超差阈值=信噪比*平均噪声

	//给定初值
	filter_mask_x=primary_x_vector;
	filter_mask_y=primary_y_vector;

	for(int itetate=0;itetate<5;itetate++)
	{
		//	根据所得x,y模板分量与估计结果的均方误差筛选出当前误差分离对象
		vector<cv::Vec2i> divorece_point;
		divorece_point.reserve(over_error_rows*over_error_cols);

		for(int i=0;i<filter_mask_rows;i++)
		{
			filter_mask_ptr=filter_mask.ptr<float>(i);

			for(int j=0;j<filter_mask_cols;j++)
			{
				float current_error=abs(filter_mask_ptr[j]-filter_mask_y[i]*filter_mask_x[j]);//误差

				if(current_error>over_error_threshold)
				{
					divorece_point.push_back(cv::Vec2i(i,j));
				}
			}
		}
		
		//	根据现有x,y向模板分量估计当前误差分离对象的剥离误差剩余值
		cv::Mat filter_mask_remains;	//剥离误差后的剩余值
		filter_mask.copyTo(filter_mask_remains);

		int divorece_point_number=divorece_point.size();
		for(int i=0;i<divorece_point_number;i++)//误差剥离项的值修改为估计值
		{
			int row=divorece_point[i][0];
			int col=divorece_point[i][1];

			filter_mask_remains.at<float>(row,col)=filter_mask_x[col]*filter_mask_y[row];
		}

		//	采用模板的基本x,y向分解算法估计x,y模板分量
		bool analyse_reult=basicFilterMaskXYAnalysis(filter_mask_remains,filter_mask_x,filter_mask_y,standard_error,flag_out);

		if(analyse_reult==false)	//程序报错
		{
			cout<<"error: in complexFilterMaskXYAnalysis"<<endl;
			return false;
		}

		if(flag_out==false)		//无法分解，降低标准差
		{
			over_error_threshold=0.8*over_error_threshold;
		}
		else
		{
			break;
		}
	}

	//8.分析分离对象
	if(flag_out==false)	//输出不可分离
	{
		return true;
	}

	delta_mask.clear();
	delta_mask.reserve(over_error_rows*over_error_cols);

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			float current_error=filter_mask_ptr[j]-filter_mask_y[i]*filter_mask_x[j];//误差

			if(abs(current_error)>standard_error+over_error_threshold)
			{
				Point_Value current_point;
				current_point.row=i;
				current_point.col=j;
				current_point.value=current_error;

				delta_mask.push_back(current_point);
			}
		}
	}

	return true;
}


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
bool basicFilterMaskXYconstAnalysis(cv::Mat& filter_mask,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& mask_const,float& standard_error,bool& flag_out)
{
	//0.尝试一次xy分解
	vector<float> filter_mask_x_z,filter_mask_y_z;
	basicFilterMaskXYAnalysis(filter_mask,filter_mask_x_z,filter_mask_y_z,standard_error,flag_out);
	if(flag_out==true)
	{
		filter_mask_x=filter_mask_x_z;
		filter_mask_y=filter_mask_y_z;
		mask_const=0;
		return true;
	}

	//1.获取filter_mask尺寸,并定义输出值大小
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	//add1(20170222)：避免计算奇异值，增加偏置量
	float add_number=1.0/(filter_mask_rows*filter_mask_cols);	//避免奇异值的偏置量
	float filter_st_mean=0;	//模板初始均值
	float* filter_mask_ptr;					//模板迭代指针

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			filter_st_mean+=filter_mask_ptr[j];
		}
	}

	if(filter_st_mean>=0)
	{
		for(int i=0;i<filter_mask_rows;i++)
		{
			filter_mask_ptr=filter_mask.ptr<float>(i);

			for(int j=0;j<filter_mask_cols;j++)
				filter_mask_ptr[j]+=add_number;
		}
	}
	else
	{
		for(int i=0;i<filter_mask_rows;i++)
		{
			filter_mask_ptr=filter_mask.ptr<float>(i);

			for(int j=0;j<filter_mask_cols;j++)
				filter_mask_ptr[j]-=add_number;
		}
	}

	//2.计算模板均值与均方
	float filter_element_average=0;			//模板元素的均值
	float filter_element_square_average=0;	//模板元素的平方均值

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			filter_element_average+=filter_mask_ptr[j];
			filter_element_square_average+=filter_mask_ptr[j]*filter_mask_ptr[j];
		}
	}
	filter_element_average=filter_element_average/(filter_mask_rows*filter_mask_cols);
	filter_element_square_average=sqrt(filter_element_square_average/(filter_mask_rows*filter_mask_cols));
	

	//3.模板0均值化
	cv::Mat filter_mask_zero_aver;	//0均值化模板
	filter_mask.convertTo(filter_mask_zero_aver,filter_mask.type());

	filter_mask_zero_aver=filter_mask_zero_aver-filter_element_average;


	//4.判断模板所有元素是否相同
	standard_error=0;
	bool same_flag=true;

	float* filter_mask_zero_aver_ptr;			//均值化模板迭代指针
	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_zero_aver_ptr=filter_mask_zero_aver.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			if(abs(filter_mask_zero_aver_ptr[j])>0.01*filter_element_square_average)
			{
				same_flag=false;
				break;
			}
			standard_error+=filter_mask_zero_aver_ptr[j]*filter_mask_zero_aver_ptr[j];
		}
		if(same_flag==false)
			break;
	}
	if(same_flag==true)	//元素全相同，输出均值
	{
		standard_error=sqrt(standard_error/(filter_mask_rows*filter_mask_cols));
		mask_const=filter_element_average;	//常数为模板均值
		filter_mask_x.clear();				//x方向输出空
		filter_mask_y.clear();				//y方向输出空

		flag_out=true;
		return true;
	}


	//5.采用基本分解计算x,y向量大致的估计值
	bool analyseed=false;
	if(abs(mask_const)<filter_element_square_average)	//常数有先验值，根据先验值计算x,y向量
	{
		cv::Mat filter_mask_2=filter_mask-mask_const;
		vector<float> filter_mask_x_z,filter_mask_y_z;
		analyseed=basicFilterMaskXYAnalysis(filter_mask_2,filter_mask_x_z,filter_mask_y_z,standard_error,flag_out);//模板的基本x,y向分解
		if(analyseed==true)
		{
			filter_mask_x=filter_mask_x_z;
			filter_mask_y=filter_mask_y_z;
		}
	}
	else
	{
		vector<float> filter_mask_x_z,filter_mask_y_z;
		analyseed=basicFilterMaskXYAnalysis(filter_mask,filter_mask_x_z,filter_mask_y_z,standard_error,flag_out);//模板的基本x,y向分解
		if(analyseed==true)
		{
			filter_mask_x=filter_mask_x_z;
			filter_mask_y=filter_mask_y_z;
		}
	}
	if(analyseed==false)
	{
		cout<<"error: -> in basicFilterMaskXYconstAnalysis"<<endl;
		return false;
	}

	//6.迭代求解xy向量
	//将初值转换为矩阵
	cv::Mat f_m_x(filter_mask_x,CV_32FC1);
	cv::Mat f_m_y(filter_mask_y,CV_32FC1);

	//迭代计算xy向量
	for(int iterate=0;iterate<4*(filter_mask_cols+filter_mask_rows);iterate++)
	{
		float rate=sqrt(standard_error/filter_element_square_average);

		//(1) 计算hx=[hy'hy*I-aver(hy)^2*E]^-1*[M'-aver(M)*E]*hy
		float y_vector_average=0;
		float y_vector_square_sum=0;

		float* fm_y_p=(float*)f_m_y.data;
		for(int i=0;i<filter_mask_rows;i++)
		{
			y_vector_average+=fm_y_p[i];
			y_vector_square_sum+=fm_y_p[i]*fm_y_p[i];
		}

		y_vector_average=y_vector_average/filter_mask_rows;

		cv::Mat function_mat_y=(y_vector_square_sum*cv::Mat::eye(filter_mask_cols,filter_mask_cols,filter_mask.type())-y_vector_average*y_vector_average);
		f_m_x=-rate*f_m_x+(1+rate)*function_mat_y.inv()*(filter_mask_zero_aver.t()*f_m_y);

		//(2) 计算hy=[hx'hx*I-aver(hx)^2*E]^-1*[M-aver(M)*E]*hx
		float x_vector_average=0;
		float x_vector_square_sum=0;

		float* fm_x_p=(float*)f_m_x.data;
		for(int i=0;i<filter_mask_cols;i++)
		{
			x_vector_average+=fm_x_p[i];
			x_vector_square_sum+=fm_x_p[i]*fm_x_p[i];
		}
		x_vector_average=x_vector_average/filter_mask_cols;

		cv::Mat function_mat_x=(x_vector_square_sum*cv::Mat::eye(filter_mask_rows,filter_mask_rows,filter_mask.type())-x_vector_average*x_vector_average);
		f_m_y=function_mat_x.inv()*(filter_mask_zero_aver*f_m_x);

		//(3)跳出条件：误差小于阈值
		float current_error=-x_vector_square_sum*y_vector_square_sum/(filter_mask_rows*filter_mask_cols)
			+x_vector_average*x_vector_average*y_vector_average*y_vector_average-filter_element_average*filter_element_average+filter_element_square_average*filter_element_square_average;
		
		if(current_error>0)
			standard_error=sqrt(current_error);
		else
			standard_error=0;

		if(standard_error<0.01*filter_element_square_average)
			break;

		
	}
	float* fm_x_p=(float*)f_m_x.data;
	for(int i=0;i<filter_mask_rows;i++)
		filter_mask_x[i]=fm_x_p[i];

	float* fm_y_p=(float*)f_m_y.data;
	for(int i=0;i<filter_mask_cols;i++)
		filter_mask_y[i]=fm_y_p[i];


	//7.根据x向量与y向量计算常数项
	float x_vector_average=0;	//x向量元素均值
	for(int i=0;i<filter_mask_cols;i++)
	{
		x_vector_average+=filter_mask_x[i];
	}
	x_vector_average=x_vector_average/filter_mask_cols;

	float y_vector_average=0;	//y向量元素均值
	for(int i=0;i<filter_mask_rows;i++)
	{
		y_vector_average+=filter_mask_y[i];
	}
	y_vector_average=y_vector_average/filter_mask_rows;

	mask_const=filter_element_average-x_vector_average*y_vector_average;


	//add_2(20170222)：从常数项中剔除偏置量，还原偏置量
	if(filter_st_mean>=0)
	{
		mask_const-=add_number;
		filter_mask=filter_mask-add_number;
	}
	else
	{
		mask_const+=add_number;
		filter_mask=filter_mask+add_number;
	}


	//8.计算均方误差
	standard_error=0;
	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			float current_error=filter_mask_ptr[j]-filter_mask_y[i]*filter_mask_x[j]-mask_const;
			standard_error+=current_error*current_error;
		}
	}
	standard_error=sqrt(standard_error/(filter_mask_rows*filter_mask_rows-filter_mask_rows-filter_mask_cols));

	if(standard_error<0.01*filter_element_square_average)
	{
		flag_out=true;
	}
	else
		flag_out=false;

	return true;
}


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
bool complexFilterMaskXYconstAnalysis(cv::Mat& filter_mask,float& gamma,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& mask_const,vector<Point_Value>& delta_mask,float& standard_error,bool& flag_out)
{
	//0.获取filter_mask尺寸,并定义输出值大小
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	//先尝试一次直接分解
	basicFilterMaskXYconstAnalysis(filter_mask,filter_mask_x,filter_mask_y,mask_const,standard_error,flag_out);
	if(flag_out==true)
	{
		delta_mask.clear();
		return true;
	}

	//1.计算各行、列之间的差分向量并归一化(包括符号归一化)
	//计算差分向量
	cv::Mat filter_mask_d_row=cv::Mat::zeros(filter_mask_rows-1,filter_mask_cols,filter_mask.type());	//行间差分
	cv::Mat filter_mask_d_col=cv::Mat::zeros(filter_mask_rows,filter_mask_cols-1,filter_mask.type());	//列间差分

	vector<int> d_row_sign(filter_mask_rows-1,0);	//各行差分的元素符号和
	vector<int> d_col_sign(filter_mask_cols-1,0);	//各列差分的元素符号和

	float *filter_mask_ptr,*filter_mask_ptr_2,*filter_mask_d_row_ptr,*filter_mask_d_col_ptr;

	for(int i=0;i<filter_mask_rows-1;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);
		filter_mask_ptr_2=filter_mask.ptr<float>(i+1);
		filter_mask_d_row_ptr=filter_mask_d_row.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			filter_mask_d_row_ptr[j]=filter_mask_ptr_2[j]-filter_mask_ptr[j];

			if(filter_mask_d_row_ptr[j]>0)
				d_row_sign[i]++;
			if(filter_mask_d_row_ptr[j]<0)
				d_row_sign[i]--;
		}
	}

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);
		filter_mask_d_col_ptr=filter_mask_d_col.ptr<float>(i);

		for(int j=0;j<filter_mask_cols-1;j++)
		{
			filter_mask_d_col_ptr[j]=filter_mask_ptr[j+1]-filter_mask_ptr[j];

			if(filter_mask_d_col_ptr[j]>0)
				d_col_sign[j]++;
			if(filter_mask_d_col_ptr[j]<0)
				d_col_sign[j]--;
		}
	}

	//向量归一化
	cv::Mat filter_mask_drow_one=cv::Mat::zeros(filter_mask_rows-1,filter_mask_cols,filter_mask.type());	//归一化行间差分
	cv::Mat filter_mask_dcol_one=cv::Mat::zeros(filter_mask_rows,filter_mask_cols-1,filter_mask.type());	//归一化列间差分

	float* filter_mask_drow_one_ptr,*filter_mask_dcol_one_ptr;

	//(1)归一化dx
	//计算dx
	vector<int> sign_element_x(filter_mask_cols,0);	//每列元素符号方向
	for(int i=0;i<filter_mask_rows-1;i++)
	{
		filter_mask_d_row_ptr=filter_mask_d_row.ptr<float>(i);
		filter_mask_drow_one_ptr=filter_mask_drow_one.ptr<float>(i);

		float drow_model=0;
		for(int j=0;j<filter_mask_cols;j++)
		{
				drow_model+=filter_mask_d_row_ptr[j]*filter_mask_d_row_ptr[j];
		}
		drow_model=sqrt(drow_model);

		for(int j=0;j<filter_mask_cols;j++)
		{
			if(drow_model==0)
			{
				filter_mask_drow_one_ptr[j]=0;
				continue;
			}

			if(d_row_sign[i]>=0)
				filter_mask_drow_one_ptr[j]=filter_mask_d_row_ptr[j]/drow_model;
			else
				filter_mask_drow_one_ptr[j]=-filter_mask_d_row_ptr[j]/drow_model;

			if(filter_mask_drow_one_ptr[j]>0)
				sign_element_x[j]++;
			if(filter_mask_drow_one_ptr[j]<0)
				sign_element_x[j]--;
		}
	}
	//校正dx
	vector<int> each_line_s_no_matched_n(filter_mask_rows-1,0);//符号不匹配个数
	for(int i=0;i<filter_mask_rows-1;i++)
	{
		filter_mask_drow_one_ptr=filter_mask_drow_one.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
				if(filter_mask_drow_one_ptr[j]*sign_element_x[j]<0)
					each_line_s_no_matched_n[i]++;
		}

		if(each_line_s_no_matched_n[i]>d_col_sign[i]/2+1)
		{
			for(int j=0;j<filter_mask_cols;j++)
				filter_mask_drow_one_ptr[j]=-filter_mask_drow_one_ptr[j];
		}
	}

	//(2)归一化dy
	vector<int> sign_element_y(filter_mask_rows,0);	//每行元素符号方向
	vector<float> dcol_model(filter_mask_cols,0);	//dy向量模

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_d_col_ptr=filter_mask_d_col.ptr<float>(i);

		for(int j=0;j<filter_mask_cols-1;j++)
		{
			dcol_model[j]+=filter_mask_d_col_ptr[j]*filter_mask_d_col_ptr[j];
		}
	}

	for(int i=0;i<filter_mask_rows;i++)
	{
		dcol_model[i]=sqrt(dcol_model[i]);
	}

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_d_col_ptr=filter_mask_d_col.ptr<float>(i);
		filter_mask_dcol_one_ptr=filter_mask_dcol_one.ptr<float>(i);

		for(int j=0;j<filter_mask_cols-1;j++)
		{
			if(dcol_model[j]==0)
			{
				filter_mask_dcol_one_ptr[j]=0;
				continue;
			}

			if(d_col_sign[j]>=0)
				filter_mask_dcol_one_ptr[j]=filter_mask_d_col_ptr[j]/dcol_model[j];
			else
				filter_mask_dcol_one_ptr[j]=-filter_mask_d_col_ptr[j]/dcol_model[j];

			if(filter_mask_dcol_one_ptr[j]>0)
				sign_element_y[i]++;
			if(filter_mask_dcol_one_ptr[j]<0)
				sign_element_y[i]--;
		}
	}
	//校正dy
	vector<int> each_col_s_no_matched_n(filter_mask_cols-1,0);//符号不匹配个数
	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_dcol_one_ptr=filter_mask_dcol_one.ptr<float>(i);

		for(int j=0;j<filter_mask_cols-1;j++)
		{
				if(filter_mask_dcol_one_ptr[j]*sign_element_y[i]<0)
					each_col_s_no_matched_n[j]++;
		}
	}
	for(int i=0;i<filter_mask_cols-1;i++)
	{
		if(each_col_s_no_matched_n[i]>d_row_sign[i]/2+1)
		{
			for(int j=0;j<filter_mask_rows;j++)
				filter_mask_dcol_one.at<float>(j,i)=-filter_mask_dcol_one.at<float>(j,i);
		}
	}
	

	//2.计算所有归一化行、列向量各个元素均值、方差
	vector<float> aver_d_row(filter_mask_cols,0);
	vector<float> std_d_row(filter_mask_cols,0);
	vector<float> aver_d_col(filter_mask_rows,0);
	vector<float> std_d_col(filter_mask_rows,0);

	//行向量均值
	for(int i=0;i<filter_mask_rows-1;i++)
	{
		filter_mask_drow_one_ptr=filter_mask_drow_one.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			aver_d_row[j]+=filter_mask_drow_one_ptr[j];
		}
	}
	for(int i=0;i<filter_mask_cols;i++)
		aver_d_row[i]=aver_d_row[i]/(filter_mask_rows-1);

	//行向量方差
	for(int i=0;i<filter_mask_rows-1;i++)
	{
		filter_mask_drow_one_ptr=filter_mask_drow_one.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			float d_drow=filter_mask_drow_one_ptr[j]-aver_d_row[j];
			std_d_row[j]+=d_drow*d_drow;
		}
	}
	for(int i=0;i<filter_mask_cols;i++)
		std_d_row[i]=sqrt(std_d_row[i]/(filter_mask_rows-1));

	//dy均值
	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_dcol_one_ptr=filter_mask_dcol_one.ptr<float>(i);

		float dcol_aver_current=0;
		for(int j=0;j<filter_mask_cols-1;j++)
		{
			dcol_aver_current+=filter_mask_dcol_one_ptr[j];
		}

		aver_d_col[i]=dcol_aver_current/(filter_mask_cols-1);
	}

	//dy方差
	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_dcol_one_ptr=filter_mask_dcol_one.ptr<float>(i);

		float dcol_std_current=0;
		for(int j=0;j<filter_mask_cols-1;j++)
		{
			float d_dcol=filter_mask_dcol_one_ptr[j]-aver_d_col[i];
			dcol_std_current+=d_dcol*d_dcol;
		}

		std_d_col[i]=sqrt(dcol_std_current/(filter_mask_cols-1));
	}


	//3.以方差倒数为权值计算加权均值向量
	vector<float> aver_w_d_row(filter_mask_cols,0);
	vector<float> aver_w_d_col(filter_mask_rows,0);

	//行加权均值
	float w_d_row=0;
	for(int i=0;i<filter_mask_rows-1;i++)
		w_d_row+=1/(std_d_col[i]+std_d_col[i+1]+0.0001);

	for(int i=0;i<filter_mask_rows-1;i++)
	{
		filter_mask_drow_one_ptr=filter_mask_drow_one.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{		
			aver_w_d_row[j]+=filter_mask_drow_one_ptr[j]/(std_d_col[i]+std_d_col[i+1]+0.0001);
		}
	}
	for(int j=0;j<filter_mask_cols;j++)
	{
		aver_w_d_row[j]=aver_w_d_row[j]/w_d_row;
	}

	//dy加权均值
	float w_d_col=0;
	for(int i=0;i<filter_mask_cols-1;i++)
		w_d_col+=1/(std_d_row[i]+std_d_row[i+1]+0.0001);

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_dcol_one_ptr=filter_mask_dcol_one.ptr<float>(i);

		float current_aver_d_col=0;
		for(int j=0;j<filter_mask_cols-1;j++)
		{	
			current_aver_d_col+=filter_mask_dcol_one_ptr[j]/(std_d_row[j]+std_d_row[j+1]+0.0001);
		}

		aver_w_d_col[i]=current_aver_d_col/w_d_col;
	}


	//4.根据向量与加权均值向量之间的误差获得分离点所在行列
	vector<bool> divide_x(filter_mask_cols,false);	//标记分离列
	vector<bool> divide_y(filter_mask_rows,false);	//标记分离行

	vector<float> x_threshold(filter_mask_cols,0);	//x阈值
	for(int i=0;i<filter_mask_cols;i++)
		x_threshold[i]=1.414*(std_d_row[i]+gamma/filter_mask_cols);

	for(int i=0;i<filter_mask_rows-1;i++)
	{
		filter_mask_drow_one_ptr=filter_mask_drow_one.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			float d_drow=filter_mask_drow_one_ptr[j]-aver_w_d_row[j];
			if(abs(d_drow)>x_threshold[j])
			{
				divide_x[j]=true;
			}
		}
	}

	vector<float> y_threshold(filter_mask_rows,0);	//y阈值
	for(int i=0;i<filter_mask_rows;i++)
		y_threshold[i]=1.414*(std_d_col[i]+gamma/filter_mask_rows);

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_dcol_one_ptr=filter_mask_dcol_one.ptr<float>(i);

		for(int j=0;j<filter_mask_cols-1;j++)
		{
			float d_d_col=filter_mask_dcol_one_ptr[j]-aver_w_d_col[i];
			if(abs(d_d_col)>y_threshold[i])
			{
				divide_y[i]=true;
				break;
			}
		}
	}

	//统计所使用的行列数
	float x_used_number=0;	//使用的列数
	float y_used_number=0;	//使用的行数

	for(int i=0;i<filter_mask_cols;i++)
	{
		if(divide_x[i]==false)
			x_used_number++;
	}
	for(int i=0;i<filter_mask_rows;i++)
	{
		if(divide_y[i]==false)
			y_used_number++;
	}

	if(x_used_number==0||y_used_number==0||filter_mask_cols/3-x_used_number+filter_mask_rows/3-y_used_number>0&&x_used_number<2&&y_used_number<2)	//所分离的行列数目过多，该分离达不到降低计算量的意义
	{
		flag_out=false;
		return true;
	}


	//5.计算x,y模板除分离点行列以外的元素初始估计值
	//计算求和式
	vector<float> x_mask_prime(filter_mask_cols,0);
	vector<float> y_mask_prime(filter_mask_rows,0);

	for(int i=0;i<filter_mask_rows-1;i++)
	{
		if(divide_y[i]==true||divide_y[i+1]==true)
			continue;

		filter_mask_drow_one_ptr=filter_mask_drow_one.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			x_mask_prime[j]+=filter_mask_drow_one_ptr[j];
		}
	}

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_dcol_one_ptr=filter_mask_dcol_one.ptr<float>(i);

		for(int j=0;j<filter_mask_cols-1;j++)
		{
			if(divide_x[j]==true||divide_x[j+1]==true)
				continue;

			y_mask_prime[i]+=filter_mask_dcol_one_ptr[j];
		}
	}

	//计算待定系数
	float sum_xm_square=0;
	float sum_dym_square=0;
	for(int i=0;i<filter_mask_cols;i++)
	{
		if(divide_x[i]==true)
			continue;

		sum_xm_square+=x_mask_prime[i]*x_mask_prime[i];
	}
	for(int i=0;i<filter_mask_rows-1;i++)
	{
		if(divide_y[i]==true||divide_y[i+1]==true)
			continue;

		float dy_mask=y_mask_prime[i+1]-y_mask_prime[i];
		sum_dym_square+=dy_mask*dy_mask;
	}

	float u_m_dxy_sum=0;//sum{[hx(i+1)-hx(i)]hy(j)*ux(i,j)}
	for(int i=0;i<filter_mask_rows-1;i++)
	{
		if(divide_y[i]==true||divide_y[i+1]==true)
			continue;

		filter_mask_d_row_ptr=filter_mask_d_row.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			if(divide_x[j]==true)
				continue;

			u_m_dxy_sum+=x_mask_prime[j]*(y_mask_prime[i+1]-y_mask_prime[i])*filter_mask_d_row_ptr[j];
		}
	}

	float lamda=u_m_dxy_sum/(sum_xm_square*sum_dym_square);


	//6.计算常数项初始估计值
	mask_const=0;
	for(int i=0;i<filter_mask_rows;i++)
	{
		if(divide_y[i]==true)
			continue;

		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			if(divide_x[j]==true)
				continue;

			mask_const+=filter_mask_ptr[j]-y_mask_prime[i]*x_mask_prime[j]*lamda;
		}
	}
	mask_const=mask_const/(x_used_number*y_used_number);


	//7.根据非分离行列，计算x,y模板完整的初始估计
	x_mask_prime=vector<float>(filter_mask_cols,0);
	y_mask_prime=vector<float>(filter_mask_rows,0);

	//统计各行列主方向
	vector<int> row_sign(filter_mask_rows);	//各行主方向
	vector<int> col_sign(filter_mask_cols);	//各列主方向

	cv::Mat filter_mask_d=filter_mask-mask_const;//模板矩阵减去常数项
	float* filter_mask_d_ptr;

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_d_ptr=filter_mask_d.ptr<float>(i);
		for(int j=0;j<filter_mask_cols;j++)
		{
			if(filter_mask_d_ptr[j]>0)
			{
				row_sign[i]++;
				col_sign[j]++;
			}
			else
			{
				if(filter_mask_d_ptr[j]<0)
				{
					row_sign[i]--;
					col_sign[j]--;
				}
			}
		}
	}

	//计算初始估计
	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_d_ptr=filter_mask_d.ptr<float>(i);
		for(int j=0;j<filter_mask_cols;j++)
		{
			//x
			if(divide_y[i]==false)
			{
				if(row_sign[i]>=0)
				{
					x_mask_prime[j]+=filter_mask_d_ptr[j];
				}
				else
				{
					x_mask_prime[j]-=filter_mask_d_ptr[j];
				}
			}

			//y
			if(divide_x[j]==false)
			{
				if(col_sign[j]>=0)
				{
					y_mask_prime[i]+=filter_mask_d_ptr[j];
				}
				else
				{
					y_mask_prime[i]-=filter_mask_d_ptr[j];
				}
			}
		}
	}

	//估计值参数归一化
	float x_mask_prime_square_sum=0;
	float y_mask_prime_square_sum=0;
	float xy_mask_sum=0;

	for(int i=0;i<filter_mask_cols;i++)
	{
		if(divide_x[i]==false)
			x_mask_prime_square_sum+=x_mask_prime[i]*x_mask_prime[i];
	}
	for(int i=0;i<filter_mask_rows;i++)
	{
		if(divide_y[i]==false)
			y_mask_prime_square_sum+=y_mask_prime[i]*y_mask_prime[i];
	}
	for(int i=0;i<filter_mask_rows;i++)
	{
		if(divide_y[i]==true)
			continue;

		filter_mask_d_ptr=filter_mask_d.ptr<float>(i);
		for(int j=0;j<filter_mask_cols;j++)
		{
			if(divide_x[j]==true)
				continue;

			xy_mask_sum+=y_mask_prime[i]*x_mask_prime[j]*filter_mask_d_ptr[j];
		}
	}
	float alpha=sqrt(xy_mask_sum/(x_mask_prime_square_sum*y_mask_prime_square_sum));


	for(int i=0;i<filter_mask_rows;i++)
	{
		y_mask_prime[i]=y_mask_prime[i]*alpha;
	}
	for(int i=0;i<filter_mask_cols;i++)
	{
		x_mask_prime[i]=x_mask_prime[i]*alpha;
	}


	//9.迭代更新模板分解
	filter_mask_x=x_mask_prime;
	filter_mask_y=y_mask_prime;

	//计算分离阈值
	float over_error_threshold=0;
	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);
		for(int j=0;j<filter_mask_cols;j++)
		{
			over_error_threshold+=filter_mask_ptr[j]*filter_mask_ptr[j];
		}	
	}
	over_error_threshold=gamma*sqrt(over_error_threshold/(filter_mask_rows*filter_mask_cols));

	for(int iterate=0;iterate<(filter_mask_rows+filter_mask_cols);iterate++)
	{
		//(1)根据hx,hy,const计算各个元素的dM,并根据阈值判断分离项
		vector<cv::Vec2i> divorece_point;
		divorece_point.reserve((filter_mask_cols-x_used_number)*(filter_mask_rows-y_used_number));

		for(int i=0;i<filter_mask_rows;i++)
		{
			filter_mask_ptr=filter_mask.ptr<float>(i);

			for(int j=0;j<filter_mask_cols;j++)
			{
				float current_error=abs(filter_mask_ptr[j]-filter_mask_y[i]*filter_mask_x[j]-mask_const);//误差

				if(current_error>over_error_threshold)
				{
					divorece_point.push_back(cv::Vec2i(i,j));
				}
			}
		}

		//(2)对于分离项，采用M(i,j)=hx(i)*hy(j)+const代替原值
		cv::Mat filter_mask_remains;	//剥离误差后的剩余值
		filter_mask.copyTo(filter_mask_remains);

		int divorece_point_number=divorece_point.size();
		for(int i=0;i<divorece_point_number;i++)//误差剥离项的值修改为估计值
		{
			int row=divorece_point[i][0];
			int col=divorece_point[i][1];

			filter_mask_remains.at<float>(row,col)=filter_mask_x[col]*filter_mask_y[row]+mask_const;
		}

		//(3)采用基本hx,hy,const分解估计分解值，及其标准差
		bool basic_result=basicFilterMaskXYconstAnalysis(filter_mask_remains,filter_mask_x,filter_mask_y,mask_const,standard_error,flag_out);

		if(basic_result==false)	//程序报错
		{
			cout<<"error: in complexFilterMaskXYconstAnalysis"<<endl;
		}

		if(flag_out==false)		//无法分解，降低标准差
		{
			over_error_threshold=0.9*over_error_threshold;
		}
		else
		{
			break;
		}
	}


	//10.分析分离对象dM
	if(flag_out==false)	//输出不可分离
	{
		return true;
	}

	delta_mask.clear();
	delta_mask.reserve((filter_mask_cols-x_used_number)*(filter_mask_rows-y_used_number));

	bool vec_flag=true;	//标记VEC项是否存在（即分离结果是否只有常数项）
	if(filter_mask_y.size()==0||filter_mask_x.size()==0)
		vec_flag=false;

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			float current_error=filter_mask_ptr[j]-mask_const;//误差

			if(vec_flag)
				current_error=current_error-filter_mask_y[i]*filter_mask_x[j];

			if(abs(current_error)>standard_error+over_error_threshold)
			{
				Point_Value current_point;
				current_point.row=i;
				current_point.col=j;
				current_point.value=current_error;

				delta_mask.push_back(current_point);
			}
		}
	}

	if(delta_mask.size()>filter_mask_rows*filter_mask_cols/4)	//分离点个数不能太多，否则分解后不会提高运算速度
	{
		flag_out=false; 
	}

	return true;
}


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
bool vectorFilterMaskSymmetryAnalysis(vector<float>& filter_mask,float& gamma,vector<float>& symmetry_mask,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out)
{
	//1.获取filter_mask尺寸,并定义输出值大小
	int filter_mask_size=filter_mask.size();	//向量长度
	int left_middle_position=filter_mask_size/2-1;	//向量中间偏左的坐标

	symmetry_mask=vector<float>(filter_mask_size,0);	//确定输出向量尺寸


	//2.计算分离阈值
	float divide_threshold=0;
	for(int i=0;i<filter_mask_size;i++)
	{
		divide_threshold+=filter_mask[i]*filter_mask[i];
	}
	divide_threshold=gamma*sqrt(divide_threshold/(filter_mask_size*2));


	//3.获得分离点
	delta_mask.clear();
	delta_mask.reserve(left_middle_position);
	vector<bool> is_delta_mask(filter_mask_size,false);	//标记各项是否为分离项

	for(int i=0;i<left_middle_position+1;i++)
	{
		float d_mask_current=filter_mask[filter_mask_size-1-i]-filter_mask[i];
		if(abs(d_mask_current)>divide_threshold)
		{
			delta_mask.push_back(cv::Vec2f(filter_mask_size-1-i+0.001,d_mask_current));
			is_delta_mask[filter_mask_size-1-i]=true;
		}
	}
	if(delta_mask.size()>left_middle_position/2)	//如果分离点过多，该分离无意义
	{
		flag_out=false;
		return true;
	}

	//4.计算输出对称模板
	//非分离项计算均值
	for(int i=0;i<left_middle_position+1;i++)
	{
		float average=(filter_mask[i]+filter_mask[filter_mask_size-1-i])/2;	

		symmetry_mask[i]=average;
		symmetry_mask[filter_mask_size-1-i]=average;
	}
	if(2*left_middle_position+2<filter_mask_size)	//模板有奇数项,给非分离模板附上中间项
	{
		symmetry_mask[left_middle_position+1]=filter_mask[left_middle_position+1];
	}

	//分离项以前半部分的值为准
	int divided_number=delta_mask.size();
	for(int i=0;i<divided_number;i++)
	{
		int position=delta_mask[i][0];	//注意此处position>left_middle_position+1
		symmetry_mask[filter_mask_size-1-position]=filter_mask[filter_mask_size-1-position];
		symmetry_mask[position]=filter_mask[filter_mask_size-1-position];
	}

	//5.计算均方误差
	standard_error=0;

	for(int i=0;i<left_middle_position+1;i++)
	{
		if(is_delta_mask[filter_mask_size-1-i]==false)	//只有非分离项存在误差
		{
			float current_error=(filter_mask[filter_mask_size-1-i]-filter_mask[i]);//error=[T(j)-T(m-1-j)]/2,但两个对称元素同误差
			standard_error+=current_error*current_error/2;
		}
	}
	standard_error=sqrt(standard_error/filter_mask_size);

	flag_out=true;
	return true;
}


/*	1.6 近似等差向量模板的分解
	输入：
	cv::Mat& filter_mask			模板向量
	float& gamma					信噪比
	输出：
	float& filter_constant			模板关于坐标的常数项
	float& filter_linear			模板关于坐标的一次项
	vector<cv::Vec2f>& delta_mask	稀疏误差向量(x,df)
	float& standard_error
	bool flag_out					输入模板是否可拆解（ture为可拆解，此时模板的x,y分量拆解有效）
	当函数返回false时，程序出错*/
bool gradeVectorFilterMaskAnalysisOriginal(vector<float>& filter_mask,float& gamma,float& filter_constant,float& filter_linear,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out)
{
	//1.获取filter_mask尺寸,并定义输出值大小
	int filter_mask_size=filter_mask.size();	//向量长度


	//2.计算分离阈值
	float divide_threshold=0;
	for(int i=0;i<filter_mask_size;i++)
	{
		divide_threshold+=filter_mask[i]*filter_mask[i];
	}
	divide_threshold=gamma*sqrt(divide_threshold/(filter_mask_size*2));


	//3.计算每一项dT(j)=T(j+1)-T(j),并获得分离点
	//计算每一项dT(j)=T(j+1)-T(j)
	vector<float> delta_filter_mask(filter_mask_size,0);

	for(int i=0;i<filter_mask_size-1;i++)
	{
		delta_filter_mask[i]=filter_mask[i+1]-filter_mask[i];
	}

	//计算dT均值以及加权均值
	float delta_filter_mask_aver=(filter_mask[filter_mask_size-1]-filter_mask[0])/(filter_mask_size-1);//均值
	float w_delta_filter_mask_aver=0;	//加权均值
	float weight=0;	//权重

	for(int i=0;i<filter_mask_size-1;i++)
	{
		float d_delta_filter_mask=delta_filter_mask[i]-delta_filter_mask_aver;
		float d_delta_filter_mask_2=d_delta_filter_mask*d_delta_filter_mask;

		w_delta_filter_mask_aver+=delta_filter_mask[i]/(d_delta_filter_mask_2+0.001);
		weight+=1/(d_delta_filter_mask_2+0.001);
	}
	w_delta_filter_mask_aver=w_delta_filter_mask_aver/weight;

	//获得分离点
	vector<float> d_delta_filter_mask(filter_mask_size-1,0);	//dT与加权均值之差
	vector<bool> divide_filter_mask(filter_mask_size,true);		//标记每个元素是否为分离点

	//计算每dT的误差
	for(int i=0;i<filter_mask_size-1;i++)
	{
		d_delta_filter_mask[i]=delta_filter_mask[i]-w_delta_filter_mask_aver;
	}

	//分析非分离的种子点 dT(j)==0&&dT(j+1)==0
	int first_seed=filter_mask_size;	//标记第一个种子点坐标
	int last_seed=-1;					//标记最后一个种子点坐标

	for(int i=0;i<filter_mask_size-2;i++)
	{
		if(abs(d_delta_filter_mask[i])<=divide_threshold)
		{
			float sum_d_delta_filter_mask=0;
			for(int j=i+1;j<filter_mask_size-1;j++)
			{
				sum_d_delta_filter_mask+=d_delta_filter_mask[j];
				if(abs(sum_d_delta_filter_mask)<=divide_threshold)
				{
					divide_filter_mask[i+1]=false;
					if(first_seed==filter_mask_size)
					{
						first_seed=i+1;
					}
					if(i+1>last_seed)
					{
						last_seed=i+1;
					}

					break;
				}
			}
		}
	}
	if(last_seed==-1)	//没有种子点，无法判断可分离性
	{
		flag_out=false;
		return true;
	}

	//根据非分离的种子点估计分离点
	for(int i=0;i<filter_mask_size;i++)
	{
		if(divide_filter_mask[i]==false)	//跳过非分离种子点
		{
			continue;
		}

		if(i<last_seed)//i之后有种子点，因此需要向后搜索
		{
			float sum_d_delta_filter=0;
			for(int j=i;j<last_seed;j++)
			{
				if(divide_filter_mask[j]==false)//第j项为d(j+1)-d(j)
					break;

				sum_d_delta_filter+=d_delta_filter_mask[j];
			}
			if(abs(sum_d_delta_filter)<=divide_threshold)
			{
				divide_filter_mask[i]=false;
			}
		}
		else	//i之后没有种子点，因此需要向前搜索
		{
			float sum_d_delta_filter=0;
			for(int j=i-1;j>=first_seed;j--)
			{
				sum_d_delta_filter+=d_delta_filter_mask[j];
				
				if(divide_filter_mask[j]==false)//第j项为d(j+1)-d(j)
					break;
			}
			if(abs(sum_d_delta_filter)<=divide_threshold)
			{
				divide_filter_mask[i]=false;
			}
		}
	}

	//统计分离点个数，若大于m/2分解无意义，跳出
	int divide_number=0;	//分解个数
	for(int i=0;i<filter_mask_size;i++)
	{
		if(divide_filter_mask[i]==true)
		{
			divide_number++;
		}
	}
	if(divide_number>filter_mask_size/2)
	{
		flag_out=false;
		return true;
	}

	//4.计算常数项a,一次项b的估计值
	float sum_j=0;					//j的和
	float sum_j_square=0;			//j^2的和
	float sum_filter_mask=0;		//T(j)的和
	float w_sum_filter_mask=0;		//j*T(j)的和

	for(int i=0;i<filter_mask_size;i++)
	{
		if(divide_filter_mask[i]==false)
		{
			sum_j+=i;
			sum_j_square+=i*i;
			sum_filter_mask+=filter_mask[i];
			w_sum_filter_mask+=i*filter_mask[i];
		}
	}

	int no_divide_number=filter_mask_size-divide_number;	//非分离项个数

	filter_linear=(no_divide_number*w_sum_filter_mask-sum_j*sum_filter_mask)/(no_divide_number*sum_j_square-sum_j*sum_j);//一次项
	filter_constant=(sum_filter_mask-filter_linear*sum_j)/no_divide_number;//常数项
	
	if(abs(filter_linear/filter_constant)<0.01)	//一次项太小，忽略不计
	{
		filter_constant=filter_constant*(0.5*filter_linear+1);
		filter_linear=0;
	}

	//5.计算分离项以及均方标准差
	delta_mask.clear();
	delta_mask.reserve(divide_number);

	standard_error=0;	//均方标准差

	for(int i=0;i<filter_mask_size;i++)
	{
		float d_filter_mask_value=filter_mask[i]-filter_constant-i*filter_linear;//当前点误差
		if(divide_filter_mask[i]==true)	//分离点
		{
			delta_mask.push_back(cv::Vec2f(i+0.001,d_filter_mask_value));
		}
		else
		{
			standard_error+=d_filter_mask_value*d_filter_mask_value;
		}
	}

	standard_error=sqrt(standard_error/(filter_mask_size-2));

	flag_out=true;
	return true;
}
bool gradeVectorFilterMaskAnalysis(vector<float>& filter_mask,float& gamma,float& filter_constant,float& filter_linear,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out)
{
	//1.获取filter_mask尺寸,并定义输出值大小
	int filter_mask_size=filter_mask.size();	//向量长度

	//2.计算分离阈值
	float divide_threshold=0;
	for(int i=0;i<filter_mask_size;i++)
	{
		divide_threshold+=filter_mask[i]*filter_mask[i];
	}
	divide_threshold=gamma*sqrt(divide_threshold/filter_mask_size);

	//3.采用newton迭代计算初步估计
	cv::Mat filter_para=cv::Mat::zeros(2,1,CV_32FC1);	//待定参数初值
	cv::Mat const_para(filter_mask,true);	//求解过程中的函数常数量

	//给定初值
	filter_linear=0;
	float half_size=(filter_mask_size+1)/2;
	for(int i=half_size;i<filter_mask_size;i++)
		filter_linear+=filter_mask[i]-filter_mask[i-half_size];

	filter_linear/=(half_size*(filter_mask_size-half_size));

	filter_constant=0;
	for(int i=0;i<filter_mask_size;i++)
		filter_constant+=filter_mask[i];

	filter_constant=filter_constant/filter_mask_size-filter_linear*(filter_mask_size-1)/2;

	filter_para.at<float>(0)=filter_constant;
	filter_para.at<float>(1)=filter_linear;

	//迭代计算
	bool lf_re=nolinearFunMinistNewtonSolver(sumSqrtFunc,divide_threshold*0.01,divide_threshold*0.01,filter_para,const_para);
	if(lf_re==false)
		return false;

	filter_constant=filter_para.at<float>(0);
	filter_linear=filter_para.at<float>(1);

	//迭代求解分离项
	vector<bool> divide_filter_mask(filter_mask_size,true);		//标记每个元素是否为分离点

	for(int iter=0;iter<4;iter++)
	{
		//4.计算拟合误差,预判分离项
		vector<bool> curr_divide_filter_mask(filter_mask_size,false);		//标记每个元素是否为分离点
		int divide_number=0;	//分解个数

		for(int i=0;i<filter_mask_size;i++)
		{
			float d_value=abs(filter_mask[i]-filter_constant-filter_linear*i);
			if(d_value>divide_threshold)
			{
				curr_divide_filter_mask[i]=true;
				divide_number++;
			}
			else
			{
				standard_error+=d_value*d_value;
			}
		}

		standard_error=sqrt(standard_error/(filter_mask_size-2));

		//若分离点个数大于m/2分解无意义，跳出
		if(divide_number>filter_mask_size/2)
		{
			flag_out=false;
			return true;
		}

		//6.当前分离项与之前对比,相同则跳出
		bool same=true;
		for(int i=0;i<filter_mask_size;i++)
		{
			if(curr_divide_filter_mask[i]!=divide_filter_mask[i])
			{
				same=false;
				break;
			}
		}

		if(same&&standard_error<divide_threshold)
		{
			break;
		}

		divide_filter_mask=curr_divide_filter_mask;

		//5.排除分离项，计算a,b估计值
		float sum_j=0;					//j的和
		float sum_j_square=0;			//j^2的和
		float sum_filter_mask=0;		//T(j)的和
		float w_sum_filter_mask=0;		//j*T(j)的和

		for(int i=0;i<filter_mask_size;i++)
		{
			if(curr_divide_filter_mask[i]==false)
			{
				sum_j+=i;
				sum_j_square+=i*i;
				sum_filter_mask+=filter_mask[i];
				w_sum_filter_mask+=i*filter_mask[i];
			}
		}

		int no_divide_number=filter_mask_size-divide_number;	//非分离项个数

		filter_linear=(no_divide_number*w_sum_filter_mask-sum_j*sum_filter_mask)/(no_divide_number*sum_j_square-sum_j*sum_j);//一次项
		filter_constant=(sum_filter_mask-filter_linear*sum_j)/no_divide_number;//常数项

		if(abs(filter_linear*filter_mask_size/filter_constant)<0.01)	//一次项太小，忽略不计
		{
			filter_constant+=filter_linear*(filter_mask_size-1)/2;
			filter_linear=0;
		}
	}
	//6.计算分离项以及均方标准差
	standard_error=0;	//均方标准差
	int divide_number=0;
	delta_mask.clear();

	for(int i=0;i<filter_mask_size;i++)
	{
		float d_filter_mask_value=filter_mask[i]-filter_constant-i*filter_linear;//当前点误差
		if(divide_filter_mask[i]==true)	//分离点
		{
			delta_mask.push_back(cv::Vec2f(i+0.001,d_filter_mask_value));
			divide_number++;
		}
		else
		{
			standard_error+=d_filter_mask_value*d_filter_mask_value;
		}
	}

	standard_error=sqrt(standard_error/(filter_mask_size-2));

	if(standard_error<divide_threshold&&divide_number<filter_mask_size/2)
	{
		flag_out=true;
	}
	else
	{
		flag_out=false;
	}
	return true;
}
/*	待求解函数 min J=sum(sqrt(|h(j)-a-b*j|))
	输入量中	a=fun_in[0]
				b=fun_in[1]*/
void sumSqrtFunc(cv::Mat& base_para,cv::Mat& fun_in,float& fun_out)
{
	int n=max(base_para.rows,base_para.cols);	//输入数据个数
	float* p_base_pare=(float*)base_para.data;
	float* p_fun_in=(float*)fun_in.data;

	fun_out=0;
	for(int j=0;j<n;j++)
		fun_out+=sqrt(abs(p_base_pare[j]-p_fun_in[0]-p_fun_in[1]*j));

}

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
bool parabolaVectorFilterMaskAnalysis(vector<float>& filter_mask,float& gamma,float& filter_constant,float& filter_linear,float& filter_p2,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out)
{
	//1.获取filter_mask尺寸,并定义输出值大小
	int filter_mask_size=filter_mask.size();	//向量长度

	//数据准备
	cv::Mat A(filter_mask_size,3,CV_32FC1);
	cv::Mat Y(filter_mask_size,1,CV_32FC1);
	float* point_A;
	float* point_Y=(float*)Y.data;
	for(int i=0;i<filter_mask_size;i++)
	{
		//A
		point_A=A.ptr<float>(i);

		point_A[0]=1;
		for(int j=1;j<3;j++)
			point_A[j]=point_A[j-1]*i;

		//Y
		point_Y[i]=filter_mask[i];
	}

	//cout<<A<<endl;//test
	//cout<<Y<<endl;//test

	cv::Mat At=A.t();
	float* point_At;

	cv::Mat ATAuse(3,3,CV_32FC1);
	cv::Mat ATYuse(3,1,CV_32FC1);
	float* point_ATYuse=(float*)ATYuse.data;


	//2.计算分离阈值
	float divide_threshold=0;
	for(int i=0;i<filter_mask_size;i++)
	{
		divide_threshold+=filter_mask[i]*filter_mask[i];
	}
	divide_threshold=gamma*sqrt(divide_threshold/filter_mask_size);

	//3.采用newton迭代计算初步估计
	cv::Mat filter_para=cv::Mat::zeros(3,1,CV_32FC1);	//待定参数初值
	cv::Mat const_para(filter_mask,true);	//求解过程中的函数常数量

	//给定初值
	int half_size=filter_mask_size/2;
	int hhalf_size=half_size/2;

	filter_p2=0;		//2次项
	for(int i=2*hhalf_size;i<filter_mask_size;i++)
		filter_p2+=filter_mask[i]+filter_mask[i-2*hhalf_size]-2*filter_mask[i-hhalf_size];

	filter_p2=filter_p2/(2*hhalf_size*hhalf_size*(filter_mask_size-2*hhalf_size));

	filter_linear=0;	//1次项
	for(int i=half_size;i<filter_mask_size;i++)
		filter_linear+=filter_mask[i]-filter_mask[i-half_size];

	filter_linear=filter_linear/(half_size*(filter_mask_size-half_size))-filter_p2*filter_mask_size;

	filter_constant=0;	//0次项
	for(int i=0;i<filter_mask_size;i++)
		filter_constant+=filter_mask[i];

	filter_constant=filter_constant/filter_mask_size-filter_linear*(filter_mask_size-1)/2-filter_p2*(filter_mask_size-1)*(filter_mask_size-1)/3;
	
	filter_para.at<float>(0)=filter_constant;
	filter_para.at<float>(1)=filter_linear;
	filter_para.at<float>(2)=filter_p2;

	//迭代计算
	bool lf_re=nolinearFunMinistNewtonSolver(sumSqrtFunc2,divide_threshold*0.01,divide_threshold*0.01,filter_para,const_para);
	if(lf_re==false)
		return false;

	filter_constant=filter_para.at<float>(0);
	filter_linear=filter_para.at<float>(1);
	filter_p2=filter_para.at<float>(2);

	//迭代求解分离项
	vector<bool> divide_filter_mask(filter_mask_size,true);		//标记每个元素是否为分离点

	for(int iter=0;iter<4;iter++)
	{
		//4.计算拟合误差,预判分离项
		vector<bool> curr_divide_filter_mask(filter_mask_size,false);		//标记每个元素是否为分离点
		int divide_number=0;	//分解个数
		standard_error=0;		//均方标准差

		for(int i=0;i<filter_mask_size;i++)
		{
			float d_value=abs(filter_mask[i]-filter_constant-filter_linear*i-filter_p2*i*i);
			if(d_value>divide_threshold)
			{
				curr_divide_filter_mask[i]=true;
				divide_number++;
			}
			else
			{
				standard_error+=d_value*d_value;
			}
		}
		standard_error=sqrt(standard_error/(filter_mask_size-2));

		//若分离点个数大于m/2分解无意义，跳出
		if(divide_number>filter_mask_size/2)
		{
			flag_out=false;
			return true;
		}


		//5.当前分离项与之前对比,相同则跳出
		bool same=true;
		for(int i=0;i<filter_mask_size;i++)
		{
			if(curr_divide_filter_mask[i]!=divide_filter_mask[i])
			{
				same=false;
				break;
			}
		}

		if(same&&standard_error<divide_threshold)
		{
			break;
		}

		divide_filter_mask=curr_divide_filter_mask;


		//6.排除分离项，计算系数估计值
		float ata,aty;
		for(int i=0;i<3;i++)
		{
			point_At=At.ptr<float>(i);
			float* point_ATA=ATAuse.ptr<float>(i);

			aty=0;
			for(int j=0;j<filter_mask_size;j++)
			{
				if(curr_divide_filter_mask[j]==false)
					aty+=point_At[j]*point_Y[j];
			}
			point_ATYuse[i]=aty;

			for(int j=0;j<3;j++)
			{
				float* point_At2=At.ptr<float>(j);
				ata=0;

				for(int k=0;k<filter_mask_size;k++)
				{
					if(curr_divide_filter_mask[k]==false)
						ata+=point_At[k]*point_At2[k];
				}

				point_ATA[j]=ata;
			}
		}

		//cout<<ATAuse<<endl;//test
		//cout<<ATYuse<<endl;//test

		cv::solve(ATAuse,ATYuse,filter_para);	//求解系数

		filter_constant=filter_para.at<float>(0);
		filter_linear=filter_para.at<float>(1);
		filter_p2=filter_para.at<float>(2);

		if(abs(filter_p2*half_size*half_size/filter_constant)<0.01)	//2次项太小，忽略不计
		{
			filter_constant+=filter_p2*(2*filter_mask_size-1)*(filter_mask_size-1)/6;
			filter_p2=0;
		}

		if(abs(filter_linear*half_size/filter_constant)<0.01)	//1次项太小，忽略不计
		{
			filter_constant+=filter_linear*(filter_mask_size-1)/2;
			filter_linear=0;
		}
	}

	//6.计算分离项以及均方标准差
	standard_error=0;	//均方标准差

	int divide_number=0;
	delta_mask.clear();
	for(int i=0;i<filter_mask_size;i++)
	{
		float d_filter_mask_value=filter_mask[i]-filter_constant-i*filter_linear-filter_p2*i*i;//当前点误差
		if(divide_filter_mask[i]==true)	//分离点
		{
			delta_mask.push_back(cv::Vec2f(i+0.001,d_filter_mask_value));
			divide_number++;
		}
		else
		{
			standard_error+=d_filter_mask_value*d_filter_mask_value;
		}
	}

	standard_error=sqrt(standard_error/(filter_mask_size-2));

	if(standard_error<divide_threshold&&divide_number<filter_mask_size/2)
	{
		flag_out=true;
	}
	else
	{
		flag_out=false;
	}
	return true;
}

void sumSqrtFunc2(cv::Mat& base_para,cv::Mat& fun_in,float& fun_out)
{
	int n=max(base_para.rows,base_para.cols);	//输入数据个数
	float* p_base_para=(float*)base_para.data;
	float* p_fun_in=(float*)fun_in.data;

	fun_out=0;
	for(int j=0;j<n;j++)
		fun_out+=sqrt(abs(p_base_para[j]-p_fun_in[0]-p_fun_in[1]*j-p_fun_in[2]*j*j));
}

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
bool integerMask(float gamma,cv::Mat& filter_mask,bool& flag_out,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,float& mask_const,vector<Point_Value>& xy_delta_mask,float& standard_error,Int_Mask_Divide& int_mask)
{
	//0.基本模板有理化
	int_mask.state=flag_out;	//标记模板可分性
	int_mask.size=cv::Vec2i(filter_mask.rows,filter_mask.cols);

	if(flag_out==false)
	{
		computeRationalMatrix(gamma,filter_mask,int_mask.int_mask,int_mask.mask_denominator);
	}

	//1.x向量有理化
	if(flag_out==true)
	{
		int_mask.vector_x.size=filter_mask_x.size;

		if(filter_mask_x.state==0)
		{
			computeRationalVector(gamma,filter_mask_x.basic_vector,int_mask.vector_x.basic_vector,int_mask.vector_x.denominator);
			int_mask.vector_x.state=0;
		}
		else if(filter_mask_x.state==1)	//对称向量
		{
			computeRationalVector(gamma,filter_mask_x.symmetry_vector,int_mask.vector_x.symmetry_vector,int_mask.vector_x.denominator);
			int_mask.vector_x.state=1;
		}
		else if(filter_mask_x.state==2)
		{
			vector<float> const_and_linear(2);
			const_and_linear[0]=filter_mask_x.filter_constant;
			const_and_linear[1]=filter_mask_x.filter_linear;

			vector<int> const_linear_out;

			computeRationalVector(gamma,const_and_linear,const_linear_out,int_mask.vector_x.denominator);
			int_mask.vector_x.state=2;

			int_mask.vector_x.filter_constant=const_linear_out[0];
			int_mask.vector_x.filter_linear=const_linear_out[1];
		}
		else if(filter_mask_x.state==3)
		{
			vector<float> const_and_linear(3);
			const_and_linear[0]=filter_mask_x.filter_constant;
			const_and_linear[1]=filter_mask_x.filter_linear;
			const_and_linear[2]=filter_mask_x.filter_p2;

			vector<int> const_linear_out;

			computeRationalVector(gamma,const_and_linear,const_linear_out,int_mask.vector_x.denominator);
			int_mask.vector_x.state=3;

			int_mask.vector_x.filter_constant=const_linear_out[0];
			int_mask.vector_x.filter_linear=const_linear_out[1];
			int_mask.vector_x.filter_p2=const_linear_out[2];
		}

		if(filter_mask_x.state>0)//向量分离项
		{
			int delta_size=filter_mask_x.delta_mask.size();
			int_mask.vector_x.delta_mask.resize(delta_size);

			for(int i=0;i<delta_size;i++)
			{
				if(filter_mask_x.delta_mask[i][1]>=0)
					int_mask.vector_x.delta_mask[i]=cv::Vec2i(filter_mask_x.delta_mask[i][0],filter_mask_x.delta_mask[i][1]*int_mask.vector_x.denominator+0.5);
				else
					int_mask.vector_x.delta_mask[i]=cv::Vec2i(filter_mask_x.delta_mask[i][0],filter_mask_x.delta_mask[i][1]*int_mask.vector_x.denominator-0.5);
			}
		}
	}


	//2.y向量有理化
	if(flag_out==true)
	{
		int_mask.vector_y.size=filter_mask_y.size;

		if(filter_mask_y.state==0)
		{
			computeRationalVector(gamma,filter_mask_y.basic_vector,int_mask.vector_y.basic_vector,int_mask.vector_y.denominator);
			int_mask.vector_y.state=0;
		}
		else if(filter_mask_y.state==1)	//对称向量
		{
			computeRationalVector(gamma,filter_mask_y.symmetry_vector,int_mask.vector_y.symmetry_vector,int_mask.vector_y.denominator);
			int_mask.vector_y.state=1;
		}
		else if(filter_mask_y.state==2)
		{
			vector<float> const_and_linear(2);
			const_and_linear[0]=filter_mask_y.filter_constant;
			const_and_linear[1]=filter_mask_y.filter_linear;

			vector<int> const_linear_out;

			computeRationalVector(gamma,const_and_linear,const_linear_out,int_mask.vector_y.denominator);
			int_mask.vector_y.state=2;

			int_mask.vector_y.filter_constant=const_linear_out[0];
			int_mask.vector_y.filter_linear=const_linear_out[1];
		}
		else if(filter_mask_y.state==3)
		{
			vector<float> const_and_linear(3);
			const_and_linear[0]=filter_mask_y.filter_constant;
			const_and_linear[1]=filter_mask_y.filter_linear;
			const_and_linear[2]=filter_mask_y.filter_p2;

			vector<int> const_linear_out;

			computeRationalVector(gamma,const_and_linear,const_linear_out,int_mask.vector_y.denominator);
			int_mask.vector_y.state=3;

			int_mask.vector_y.filter_constant=const_linear_out[0];
			int_mask.vector_y.filter_linear=const_linear_out[1];
			int_mask.vector_y.filter_p2=const_linear_out[2];
		}

		if(filter_mask_y.state>0)//向量分离项
		{
			int delta_size=filter_mask_y.delta_mask.size();
			int_mask.vector_y.delta_mask.resize(delta_size);

			for(int i=0;i<delta_size;i++)
			{
				int_mask.vector_y.delta_mask[i]=cv::Vec2i(filter_mask_y.delta_mask[i][0],filter_mask_y.delta_mask[i][1]*int_mask.vector_y.denominator+0.5);
			}
		}
	}


	//3.常数项有理化(采用辗转相除法)
	realNumRational(gamma,mask_const,int_mask.mask_const[0],int_mask.mask_const[1]);


	//4.分离项有理化
	int xy_delta_mask_size=xy_delta_mask.size();
	int_mask.xy_delta_mask.resize(xy_delta_mask_size);

	for(int i=0;i<xy_delta_mask_size;i++)
	{
		int_mask.xy_delta_mask[i][0]=xy_delta_mask[i].col;
		int_mask.xy_delta_mask[i][1]=xy_delta_mask[i].row;
		realNumRational(gamma,xy_delta_mask[i].value,int_mask.xy_delta_mask[i][2],int_mask.xy_delta_mask[i][3]);
	}


	//5.计算标准差
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	if(flag_out==false)	//模板不可拆解
	{
		standard_error=standard_error*standard_error*filter_mask_rows*filter_mask_cols;//展开标准差

		int filter_mask_rows=filter_mask.rows;
		int filter_mask_cols=filter_mask.cols;

		float* filter_mask_ptr;
		int* output_mat_ptr;

		for(int i=0;i<filter_mask_rows;i++)
		{
			filter_mask_ptr=filter_mask.ptr<float>(i);
			output_mat_ptr=int_mask.int_mask.ptr<int>(i);

			for(int j=0;j<filter_mask_cols;j++)
			{
				float current_error=filter_mask_ptr[j]-float(output_mat_ptr[j])/int_mask.mask_denominator;

				standard_error+=current_error*current_error;
			}
		}

		standard_error=sqrt(standard_error/(filter_mask_rows*filter_mask_cols));
	}

	if(flag_out==true)
	{
		standard_error=standard_error*standard_error*filter_mask_rows*filter_mask_cols;//展开标准差
		float x_add_error=0;
		float y_add_error=0;

		//x向量
		vector<float> x_error(filter_mask_x.basic_vector.size(),0);

		if(filter_mask_x.state==0)
		{
			int vec_size=filter_mask_x.basic_vector.size();
			for(int i=0;i<vec_size;i++)
			{
				x_error[i]=filter_mask_x.basic_vector[i]-float(int_mask.vector_x.basic_vector[i])/int_mask.vector_x.denominator;
			}
		}

		if(filter_mask_x.state==1)	//对称向量
		{
			int vec_size=filter_mask_x.basic_vector.size();
			for(int i=0;i<vec_size;i++)
			{
				x_error[i]=filter_mask_x.basic_vector[i]-float(int_mask.vector_x.symmetry_vector[i])/int_mask.vector_x.denominator;
			}
		}

		if(filter_mask_x.state==2)
		{
			int vec_size=filter_mask_x.basic_vector.size();
			for(int i=0;i<vec_size;i++)
			{
				x_error[i]=filter_mask_x.basic_vector[i]-float(int_mask.vector_x.filter_constant+int_mask.vector_x.filter_linear*i)/int_mask.vector_x.denominator;
			}
		}

		if(filter_mask_x.state>0)
		{
			int vec_size=filter_mask_x.delta_mask.size();
			for(int i=0;i<vec_size;i++)
			{
				x_error[int_mask.vector_x.delta_mask[i][0]]=x_error[int_mask.vector_x.delta_mask[i][0]]-float(int_mask.vector_x.delta_mask[i][1])/int_mask.vector_x.denominator;
			}

			for(int i=0;i<filter_mask_x.basic_vector.size();i++)
			{
				x_add_error+=x_error[i]*x_error[i];
			}
		}
		int_mask.vector_x.standard_error=sqrt(x_add_error/int_mask.vector_x.size);	//x向量有理化方差

		//y向量
		vector<float> y_error(filter_mask_y.basic_vector.size(),0);

		if(filter_mask_y.state==0)
		{
			int vec_size=filter_mask_y.basic_vector.size();
			for(int i=0;i<vec_size;i++)
			{
				y_error[i]=filter_mask_y.basic_vector[i]-float(int_mask.vector_y.basic_vector[i])/int_mask.vector_y.denominator;
			}
		}

		if(filter_mask_y.state==1)	//对称向量
		{
			int vec_size=filter_mask_y.basic_vector.size();
			for(int i=0;i<vec_size;i++)
			{
				y_error[i]=filter_mask_y.basic_vector[i]-float(int_mask.vector_y.symmetry_vector[i])/int_mask.vector_y.denominator;
			}
		}

		if(filter_mask_y.state==2)
		{
			int vec_size=filter_mask_y.basic_vector.size();
			for(int i=0;i<vec_size;i++)
			{
				y_error[i]=filter_mask_y.basic_vector[i]-float(int_mask.vector_y.filter_constant+int_mask.vector_y.filter_linear*i)/int_mask.vector_y.denominator;
			}
		}

		if(filter_mask_y.state>0)
		{
			int vec_size=filter_mask_y.delta_mask.size();
			for(int i=0;i<vec_size;i++)
			{
				y_error[int_mask.vector_y.delta_mask[i][0]]=y_error[int_mask.vector_y.delta_mask[i][0]]-float(int_mask.vector_y.delta_mask[i][1])/int_mask.vector_y.denominator;
			}

			for(int i=0;i<filter_mask_y.basic_vector.size();i++)
			{
				y_add_error+=y_error[i]*y_error[i];
			}
		}
		int_mask.vector_y.standard_error=sqrt(y_add_error/int_mask.vector_y.size);	//y向量有理化方差

		standard_error+=x_add_error*y_add_error;

		standard_error=sqrt(standard_error/(filter_mask_rows*filter_mask_cols));
	}

	//6.整理有理化结果，输出
	int_mask.standard_error=standard_error;

	return true;
}


/*	模板快速运算测试函数*/
bool filterMaskTest()
{
	//1 模板分解
	cv::Mat filter_mask=0*cv::Mat::ones(51,51,CV_32FC1)/4000;
	float sum_kernel;
	sum_kernel=0;
	for(int i=0;i<51;i++)
	{
		for(int j=0;j<51;j++)
		{
			filter_mask.at<float>(i,j)=100+(i-20)*(j-20);
			sum_kernel+=filter_mask.at<float>(i,j);
		}
	}
	filter_mask=filter_mask/sum_kernel;

	vector<float> filter_mask_x,filter_mask_y;
	float standard_error;
	bool flag_out;
	bool analyseed=basicFilterMaskXYAnalysis(filter_mask,filter_mask_x,filter_mask_y,standard_error, flag_out);//模板的基本x,y向分解

	if(analyseed==false)
		return false;

	vector<Point_Value> delta_mask;
	float gamma=0.2;	//信噪比
	analyseed=complexFilterMaskXYAnalysis(filter_mask,gamma,filter_mask_x,filter_mask_y,delta_mask,standard_error,flag_out);//复杂模板的x,y向分解

	if(analyseed==false)
		return false;

	vector<float> vector_mask(5,1);vector_mask[0]=2;vector_mask[3]=1.02;
	vector<float> symmetry_mask;
	vector<cv::Vec2f> vector_delta_mask;
	analyseed=vectorFilterMaskSymmetryAnalysis(vector_mask,gamma,symmetry_mask,vector_delta_mask,standard_error,flag_out);

	if(analyseed==false)
		return false;

	for(int i=0;i<5;i++)
		vector_mask[i]=i+1;

	vector_mask[1]=1;
	float filter_constant,filter_linear;
	analyseed=gradeVectorFilterMaskAnalysis(vector_mask,gamma,filter_constant,filter_linear,vector_delta_mask,standard_error,flag_out);
	//analyseed=gradeVectorFilterMaskAnalysis2(vector_mask,gamma,filter_constant,filter_linear,vector_delta_mask,standard_error,flag_out);
	if(analyseed==false)
		return false;

	for(int i=0;i<5;i++)
		vector_mask[i]=0.2*i*i+i;
	vector_mask[2]=4;

	float filter_p2;
	parabolaVectorFilterMaskAnalysis(vector_mask,gamma,filter_constant,filter_linear,filter_p2,vector_delta_mask,standard_error,flag_out);
	double t_use0=(double)cv::getTickCount();

	vector<float> filter_p2vec(51);
	for(int i=0;i<51;i++)
		filter_p2vec[i]=0.01*(i-25)*(i-25);
	parabolaVectorFilterMaskAnalysis(filter_p2vec,gamma,filter_constant,filter_linear,filter_p2,vector_delta_mask,standard_error,flag_out);
	t_use0=((double)cv::getTickCount()-t_use0)/cv::getTickFrequency();
	cout<<"模板抛物线分解用时:"<<t_use0<<"s"<<endl;

	Vector_Mask new_filter_mask_x,new_filter_mask_y;
	vector<Point_Value> xy_delta_mask;
	analyseed=filterMaskAnalysis(filter_mask,gamma,new_filter_mask_x,new_filter_mask_y,xy_delta_mask,standard_error,flag_out);

	//模板x,y,const分解
	float mask_const;
	analyseed=basicFilterMaskXYconstAnalysis(filter_mask,filter_mask_x,filter_mask_y, mask_const,standard_error,flag_out);

	//complexFilterMaskXYconstAnalysis(filter_mask,gamma,filter_mask_x,filter_mask_y, mask_const,delta_mask,standard_error,flag_out);

	double t_start=(double)cv::getTickCount();
	analyseed=filterMaskAnalysis(filter_mask, gamma,new_filter_mask_x,new_filter_mask_y, mask_const, xy_delta_mask, standard_error, flag_out);
	double t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();;
	cout<<"模板分解用时:"<<t_used<<"s"<<endl;

	//模板整形化
	t_start=(double)cv::getTickCount();
	Int_Mask_Divide int_mask;
	integerMask(gamma,filter_mask,flag_out,new_filter_mask_x,new_filter_mask_y,mask_const,xy_delta_mask,standard_error,int_mask);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();;
	cout<<"模板整形化用时:"<<t_used<<"s"<<endl;

	cout<<"本次模板分析测试成功！"<<endl;
	return true;
}