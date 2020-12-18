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

#include "filterAnalysis.h"

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
bool filterMaskAnalysis(cv::Mat& filter_mask,float& gamma,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,vector<Point_Value>& xy_delta_mask,float& standard_error,bool& flag_out)
{
	//1.��ȡfilter_mask�ߴ�,���������ֵ��С
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	vector<float> vector_mask_y(filter_mask_rows,0);	//y���򣨺�������ģ��
	vector<float> vector_mask_x(filter_mask_cols,0);	//x���򣨺�������ģ��


	//2.ģ��x,y��ֽ�
	vector<Point_Value> xy_analyse_delta_mask;	//x,y��ֽ�����
	float xy_standard_error;					//x,y��ֽ�������
	bool xy_flag_out;							//ģ���Ƿ����x,y��ֽ�
	bool complex_reult=complexFilterMaskXYAnalysis(filter_mask,gamma,vector_mask_x,vector_mask_y,xy_analyse_delta_mask,xy_standard_error,xy_flag_out);
	
	filter_mask_x.basic_vector=vector_mask_x;	//x��������ֽ�
	filter_mask_y.basic_vector=vector_mask_y;	//y��������ֽ�
	filter_mask_x.size=vector_mask_x.size();
	filter_mask_y.size=vector_mask_y.size();
	filter_mask_x.state=0;
	filter_mask_y.state=0;
	xy_delta_mask=xy_analyse_delta_mask;		//���x,y��ֽ�����

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

	flag_out=true;	//���ˣ�ģ�������ǿɷֽ��


	//3.����x,y��ģ��ĵȲ�ֽ�
	//x��ģ��
	float x_filter_constant,x_filter_linear;	//ģ��ĳ����һ����
	vector<cv::Vec2f> x_grade_delta_mask;		//ϡ��ֽ�����
	float x_grade_standard_error;				//ģ��ֽ�ľ�����׼��
	bool x_grade_flag_out;						//ģ���Ƿ�ɲ��

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

	//y��ģ��
	float y_filter_constant,y_filter_linear;	//ģ��ĳ����һ����
	vector<cv::Vec2f> y_grade_delta_mask;		//ϡ��ֽ�����
	float y_grade_standard_error;				//ģ��ֽ�ľ�����׼��
	bool y_grade_flag_out;						//ģ���Ƿ�ɲ��

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


	//4.����x,y��ģ��Ķ��ηֽ�
	//x��ģ��
	float x_filter_p2;	//ģ��ĳ����һ����
	float x_p2_standard_error;				//ģ��ֽ�ľ�����׼��
	bool x_p2_flag_out;						//ģ���Ƿ�ɲ��
	vector<cv::Vec2f> x_p2_delta_mask;		//ϡ��ֽ�����

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

	//y��ģ��
	float y_filter_p2;	//ģ��ĳ����һ����
	float y_p2_standard_error;				//ģ��ֽ�ľ�����׼��
	bool y_p2_flag_out;						//ģ���Ƿ�ɲ��
	vector<cv::Vec2f> y_p2_delta_mask;		//ϡ��ֽ�����

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


	//5.����x,y��ģ��ĶԳƷֽ�
	//x��ģ��
	bool symmetry_result_x=false;
	vector<float> x_symmetry_mask;				//�Գƻ�����
	vector<cv::Vec2f> x_symmetry_delta_mask;	//ϡ��ֽ�����
	float x_symmetry_standard_error;			//ģ��ֽ�ľ�����׼��
	bool x_symmetry_flag_out;					//ģ���Ƿ�ɲ��

	if(x_grade_flag_out==false&&x_p2_flag_out==false)		//�����Ȳ����λ�ʧ��ʱ���жԳƻ�����
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

	//y��ģ��
	bool symmetry_result_y=false;
	vector<float> y_symmetry_mask;				//�Գƻ�����
	vector<cv::Vec2f> y_symmetry_delta_mask;	//ϡ��ֽ�����
	float y_symmetry_standard_error;			//ģ��ֽ�ľ�����׼��
	bool y_symmetry_flag_out;					//ģ���Ƿ�ɲ��

	if(y_grade_flag_out==false&&y_p2_flag_out==false)		//�����Ȳ���λ�ʧ��ʱ���жԳƻ�����
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


	//6.�����ۺ������ģ��ֽ���
	//�����ۺ����  total_s^2=xy_s^2+x_s^2*|Tym|^2/m+y_s^2*|Txm|^2/m+x_s^2*y_s^2;
	standard_error=xy_standard_error*xy_standard_error;

	if(filter_mask_x.state>0)	//��error[Tx]!=0ʱ������|Tym|
	{
		float vector_mask_y_square=0;
		for(int i=0;i<filter_mask_rows;i++)
		{
			vector_mask_y_square+=vector_mask_y[i]*vector_mask_y[i];
		}

		standard_error=standard_error+filter_mask_x.standard_error*vector_mask_y_square/filter_mask_rows;
	}

	if(filter_mask_y.state>0)	//��error[Ty]!=0ʱ������|Txm|
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
bool filterMaskAnalysis(cv::Mat& filter_mask,float& gamma,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,
	float& mask_const,vector<Point_Value>& xy_delta_mask,float& standard_error,bool& flag_out)			//��������ʽ����
{
	//1.��ȡfilter_mask�ߴ�,���������ֵ��С
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	vector<float> vector_mask_y(filter_mask_rows,0);	//y���򣨺�������ģ��
	vector<float> vector_mask_x(filter_mask_cols,0);	//x���򣨺�������ģ��


	//2.ģ��x,y��ֽ�
	vector<Point_Value> xy_analyse_delta_mask;	//x,y��ֽ�����
	float xy_standard_error;					//x,y��ֽ�������
	bool xy_flag_out;							//ģ���Ƿ����x,y��ֽ�

	//(1)ģ��xy�ֽ�
	mask_const=0;	//������ֵΪ0
	bool complex_reult=complexFilterMaskXYAnalysis(filter_mask,gamma,vector_mask_x,vector_mask_y,xy_analyse_delta_mask,xy_standard_error,xy_flag_out);
	
	filter_mask_x.basic_vector=vector_mask_x;	//x��������ֽ�
	filter_mask_y.basic_vector=vector_mask_y;	//y��������ֽ�
	filter_mask_x.size=vector_mask_x.size();
	filter_mask_y.size=vector_mask_y.size();
	filter_mask_x.state=0;
	filter_mask_y.state=0;
	xy_delta_mask=xy_analyse_delta_mask;		//���x,y��ֽ�����

	if(xy_flag_out==false)	//ģ��xy�ֽ�ʧ�ܣ�ִ��xy,const�ֽ�
	{
		//(2)ģ��xy,const�ֽ�
		complex_reult=complexFilterMaskXYconstAnalysis(filter_mask,gamma,vector_mask_x,vector_mask_y,mask_const,xy_analyse_delta_mask,xy_standard_error,xy_flag_out);

		filter_mask_x.basic_vector=vector_mask_x;	//x��������ֽ�
		filter_mask_y.basic_vector=vector_mask_y;	//y��������ֽ�
		filter_mask_x.size=vector_mask_x.size();
		filter_mask_y.size=vector_mask_y.size();
		filter_mask_x.state=0;
		filter_mask_y.state=0;
		xy_delta_mask=xy_analyse_delta_mask;		//���x,y��ֽ�����

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

	flag_out=true;	//���ˣ�ģ�������ǿɷֽ��


	//3.����x,y��ģ��ĵȲ�ֽ�
	//�ж�x,y�����Ƿ���Ч
	if(filter_mask_x.basic_vector.size()==0||filter_mask_y.basic_vector.size()==0)	//�ֽ���ֻ�г������������
	{
		return true;
	}

	//x��ģ��
	float x_filter_constant,x_filter_linear;	//ģ��ĳ����һ����
	vector<cv::Vec2f> x_grade_delta_mask;		//ϡ��ֽ�����
	float x_grade_standard_error;				//ģ��ֽ�ľ�����׼��
	bool x_grade_flag_out;						//ģ���Ƿ�ɲ��

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

	//y��ģ��
	float y_filter_constant,y_filter_linear;	//ģ��ĳ����һ����
	vector<cv::Vec2f> y_grade_delta_mask;		//ϡ��ֽ�����
	float y_grade_standard_error;				//ģ��ֽ�ľ�����׼��
	bool y_grade_flag_out;						//ģ���Ƿ�ɲ��

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


	//4.����x,y��ģ��Ķ��ηֽ�
	//x��ģ��
	float x_filter_p2;	//ģ��ĳ����һ����
	float x_p2_standard_error;				//ģ��ֽ�ľ�����׼��
	bool x_p2_flag_out;						//ģ���Ƿ�ɲ��
	vector<cv::Vec2f> x_p2_delta_mask;		//ϡ��ֽ�����

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

	//y��ģ��
	float y_filter_p2;	//ģ��ĳ����һ����
	float y_p2_standard_error;				//ģ��ֽ�ľ�����׼��
	bool y_p2_flag_out;						//ģ���Ƿ�ɲ��
	vector<cv::Vec2f> y_p2_delta_mask;		//ϡ��ֽ�����

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


	//5.����x,y��ģ��ĶԳƷֽ�
	//x��ģ��
	bool symmetry_result_x=false;
	vector<float> x_symmetry_mask;				//�Գƻ�����
	vector<cv::Vec2f> x_symmetry_delta_mask;	//ϡ��ֽ�����
	float x_symmetry_standard_error;			//ģ��ֽ�ľ�����׼��
	bool x_symmetry_flag_out;					//ģ���Ƿ�ɲ��

	if(x_p2_flag_out==false)					//�����Ȳʧ��ʱ���жԳƻ�����
	{
		symmetry_result_x=vectorFilterMaskSymmetryAnalysis(vector_mask_x,gamma,x_symmetry_mask,
		x_symmetry_delta_mask,x_symmetry_standard_error,x_symmetry_flag_out);

		filter_mask_x.symmetry_vector=x_symmetry_mask;
		filter_mask_x.delta_mask=x_symmetry_delta_mask;
		filter_mask_x.standard_error=x_symmetry_standard_error;
		filter_mask_x.state=1;
	}

	//y��ģ��
	bool symmetry_result_y=false;
	vector<float> y_symmetry_mask;				//�Գƻ�����
	vector<cv::Vec2f> y_symmetry_delta_mask;	//ϡ��ֽ�����
	float y_symmetry_standard_error;			//ģ��ֽ�ľ�����׼��
	bool y_symmetry_flag_out;					//ģ���Ƿ�ɲ��

	if(y_p2_flag_out==false)					//�����Ȳʧ��ʱ���жԳƻ�����
	{
		symmetry_result_y=vectorFilterMaskSymmetryAnalysis(vector_mask_y,gamma,y_symmetry_mask,
		y_symmetry_delta_mask,y_symmetry_standard_error,y_symmetry_flag_out);

		filter_mask_y.symmetry_vector=y_symmetry_mask;
		filter_mask_y.delta_mask=y_symmetry_delta_mask;
		filter_mask_y.standard_error=y_symmetry_standard_error;
		filter_mask_y.state=1;
	}


	//6.�����ۺ������ģ��ֽ���
	//�����ۺ����  total_s^2=xy_s^2+x_s^2*|Tym|^2/m+y_s^2*|Txm|^2/m+x_s^2*y_s^2;
	standard_error=xy_standard_error*xy_standard_error;

	if(filter_mask_x.state>0)	//��error[Tx]!=0ʱ������|Tym|
	{
		float vector_mask_y_square=0;
		for(int i=0;i<filter_mask_rows;i++)
		{
			vector_mask_y_square+=vector_mask_y[i]*vector_mask_y[i];
		}

		standard_error=standard_error+filter_mask_x.standard_error*vector_mask_y_square/filter_mask_rows;
	}

	if(filter_mask_y.state>0)	//��error[Ty]!=0ʱ������|Txm|
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


/*	1.1 ģ��Ļ���x,y��ֽ�	
	���룺
	cv::Mat& filter_mask				ģ�����
	�����
	vector<float>& filter_mask_x		x���򣨺�������ģ��
	vector<float>& filter_mask_y		y������������ģ��
	float& standard_error				������׼��
	bool flag_out						����ģ���Ƿ�ɲ�⣨tureΪ�ɲ�⣬��ʱģ���x,y���������Ч��
	����������falseʱ���������*/
bool basicFilterMaskXYAnalysis(cv::Mat& filter_mask,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& standard_error,bool& flag_out)
{
	//1.��ȡfilter_mask�ߴ�,���������ֵ��С
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;


	//2.����filter_mask����Ԫ�ص�ƽ���ͣ�x,yģ���һ������
	float* filter_mask_ptr;		//ģ�����ָ��
	float element_square_sum=0;	//ģ��Ԫ�ص�ƽ����

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
	float xy_square_sum=x_square_sum*y_square_sum;	//x,y��ƽ����
		
	float alpha=sqrt(element_square_sum);	//x,yģ���һ������
	if(xy_square_sum>0.6*element_square_sum)
	{
		alpha=sqrt(xy_square_sum);
	}

	float total_error=0;//�ܵ�ƽ����� sum[(T(i,j)-Tx(i)Ty(j))^2]

	for(int iterate=0;iterate<2;iterate++)	//����alpha�ĳ�ֵ���ڼ���standard_error=0����Ҫ����У��alpha
	{
		//3.����yģ�������㷽�� (alpha^2*I-T'T)y=0 �о��� B=(alpha^2*I-T'T) ��Cholesky�ֽ�
		cv::Mat symmetry_matrix=cv::Mat::zeros(filter_mask_rows,filter_mask_rows,CV_32F);	//���ֽ������Գƾ���
		for(int i=0;i<filter_mask_rows;i++)
		{
			symmetry_matrix.at<float>(i,i)=element_square_sum;
		}
		symmetry_matrix=symmetry_matrix-filter_mask*filter_mask.t();

		cv::Mat down_triangle_matrix;	//�ֽ���(��������)

		bool analyse_result=normalCholeskyAnalyze(symmetry_matrix,down_triangle_matrix);//����Cholesky�ֽ�
		if(analyse_result==false)
		{
			cout<<"error: -> in basicFilterMaskXYAnalysis"<<endl;
			return false;
		}


		//4.����Cholesky�ֽ�����x,yģ�壺L'y=0,y'y=alpha
		//��ʼ����x,y��ģ��
		filter_mask_y=vector<float>(filter_mask_rows,0);
		filter_mask_x=vector<float>(filter_mask_cols,0);

		//����yģ��
		filter_mask_y[filter_mask_rows-1]=1;	//����һ���ο���ֵ

		cv::Mat down_triangle_matrix_t=down_triangle_matrix.t();
		float* down_triangle_matrix_t_ptr;	//L'�������ָ��

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


		//yģ���һ��
		float current_y_model=0;	//����y��ģ��
		for(int i=0;i<filter_mask_rows;i++)
		{
			current_y_model+=filter_mask_y[i]*filter_mask_y[i];
		}
		current_y_model=sqrt(current_y_model);

		float new_y_model=sqrt(alpha);
		for(int i=0;i<filter_mask_rows;i++)	//��һ��
		{
			filter_mask_y[i]=filter_mask_y[i]*new_y_model/current_y_model;
		}

		//����xģ��
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

		//4.�������ģ��Ը���Ԫ�ط���֮��
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

		//У��alpha�Ĺ���ֵ
		alpha=sqrt(element_square_sum-total_error/(1-(filter_mask_rows+filter_mask_cols)/(filter_mask_rows*filter_mask_cols)));
	}
	

	//5.����Ԫ�ط���֮���ж�ģ��Ŀɷֽ���
	//����Ŀ�꺯��
	float aver_standard=total_error/(filter_mask_rows*filter_mask_cols-filter_mask_rows-filter_mask_cols);
	standard_error=sqrt(aver_standard);	//���������׼����

	//������ֵ����ֵ��ģ��Ԫ��ƽ���;���
	float threshold=0.01*sqrt(element_square_sum/(filter_mask_rows*filter_mask_cols));

	//���бȽ�
	flag_out=false;
	if(aver_standard<threshold)
	{
		flag_out=true;
	}

	return true;
}


/*	1.2 ����ģ���x,y��ֽ�	
	���룺
	cv::Mat& filter_mask				ģ�����
	float& gamma						�����
	�����
	vector<float>& filter_mask_x		x���򣨺�������ģ��
	vector<float>& filter_mask_y		y������������ģ��
	vector<Point_Value>& delta_mask		ϡ��������(x,y,df)
	float& standard_error				������׼��
	bool flag_out						����ģ���Ƿ�ɲ�⣨tureΪ�ɲ�⣬��ʱģ���x,y���������Ч��
	����������falseʱ���������*/
bool complexFilterMaskXYAnalysis(cv::Mat& filter_mask,float& gamma,vector<float>& filter_mask_x,vector<float>& filter_mask_y,vector<Point_Value>& delta_mask,float& standard_error,bool& flag_out)
{
	//1.��ȡfilter_mask�ߴ�,���������ֵ��С
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	filter_mask_y=vector<float>(filter_mask_rows,0);
	filter_mask_x=vector<float>(filter_mask_cols,0);

	//�ȳ���һ��ֱ�ӷֽ�
	basicFilterMaskXYAnalysis(filter_mask,filter_mask_x,filter_mask_y,standard_error,flag_out);
	if(flag_out==true)
	{
		standard_error=standard_error;
		delta_mask.clear();
		return true;
	}


	//2.���������С��е�����ģ��
	vector<float> each_x_vector_model(filter_mask_rows,0);	//����x����ģ��
	vector<float> each_y_vector_model(filter_mask_cols,0);	//����y����ģ��

	vector<int> each_x_vector_sign(filter_mask_rows,0);	//����x������Ԫ�ط��ź�
	vector<int> each_y_vector_sign(filter_mask_cols,0);	//����y������Ԫ�ط��ź�

	float sum_element_sqaure=0;	//Ԫ��ƽ����

	float* filter_mask_ptr;		//ģ�����ָ��

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			float square_value=filter_mask_ptr[j]*filter_mask_ptr[j];//Ԫ�ص�ƽ��
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


	//3.���������С��еĹ�һ������
	cv::Mat normalized_x_vectors;	//x�����һ������
	cv::Mat normalized_y_vectors;	//y�����һ������

	filter_mask.copyTo(normalized_x_vectors);
	filter_mask.copyTo(normalized_y_vectors);

	float* normalized_x_vectors_ptr;		//x�����һ����������ָ��
	float* normalized_y_vectors_ptr;		//y�����һ��������ָ��

	for(int i=0;i<filter_mask_rows;i++)
	{
		normalized_x_vectors_ptr=normalized_x_vectors.ptr<float>(i);
		normalized_y_vectors_ptr=normalized_y_vectors.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			normalized_x_vectors_ptr[j]=normalized_x_vectors_ptr[j]/each_x_vector_model[i];
			normalized_y_vectors_ptr[j]=normalized_y_vectors_ptr[j]/each_y_vector_model[j];

			if(each_x_vector_sign[i]<0)//��һ���������������ڵ�һ���ޣ�����0��
			{
				normalized_x_vectors_ptr[j]=-normalized_x_vectors_ptr[j];
			}
			if(each_y_vector_sign[j]<0)//��һ���������������ڵ�һ���ޣ�����0��
			{
				normalized_y_vectors_ptr[j]=-normalized_y_vectors_ptr[j];
			}
		}
	}

	//4.����ÿ�С��еĹ�һ�������Ĺ��ƽ��������ÿ�С��и���Ԫ�ؾ�ֵ�Լ���׼��
	//�����ֵ
	vector<float> element_sum_in_x_vectors(filter_mask_cols,0);	//x�����и���Ԫ�ؾ�ֵ
	vector<float> element_sum_in_y_vectors(filter_mask_rows,0);	//y�����и���Ԫ�ؾ�ֵ

	vector<float> element_aver_in_x_vectors(filter_mask_cols,0);	//x�����и���Ԫ�ؾ�ֵ
	vector<float> element_aver_in_y_vectors(filter_mask_rows,0);	//y�����и���Ԫ�ؾ�ֵ

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

	//���㷽��
	vector<float> element_standard_in_x_vectors(filter_mask_cols,0);	//x�����и���Ԫ�ر�׼��
	vector<float> element_standard_in_y_vectors(filter_mask_rows,0);	//y�����и���Ԫ�ر�׼��

	cv::Mat normalized_x_vectors_error(filter_mask_rows,filter_mask_cols,CV_32F);	//x�����һ���������
	cv::Mat normalized_y_vectors_error(filter_mask_rows,filter_mask_cols,CV_32F);	//y�����һ���������

	for(int i=0;i<filter_mask_rows;i++)
	{
		normalized_x_vectors_ptr=normalized_x_vectors.ptr<float>(i);
		normalized_y_vectors_ptr=normalized_y_vectors.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			float dx_element=normalized_x_vectors_ptr[j]-element_aver_in_x_vectors[j];	//xԪ�����
			float dy_element=normalized_y_vectors_ptr[j]-element_aver_in_y_vectors[i];	//yԪ�����

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

	//�����׼�����Ȩ��ֵ
	vector<float> element_w_aver_in_x_vectors(filter_mask_cols,0);	//x�����и���Ԫ�ؼ�Ȩ��ֵ
	vector<float> element_w_aver_in_y_vectors(filter_mask_rows,0);	//y�����и���Ԫ�ؼ�Ȩ��ֵ

	float weight_x=0;//����x����Ȩ��
	float weight_y=0;//����y����Ȩ��

	for(int i=0;i<filter_mask_rows;i++)//�������x����Ȩ��
	{
		if(element_standard_in_y_vectors[i]<0.00001)
			weight_x+=100;
		else
			weight_x+=1/element_standard_in_y_vectors[i];
	}

	for(int i=0;i<filter_mask_cols;i++)//�������y����Ȩ��
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


	//5.��¼�����Ȩ��ֵ֮�����2����׼���x��y����Ԫ�ض�Ӧ���С������꣬�������ظ�ͳ�ƣ��õ�Ǳ�����������
	
	//�������С����Ƿ񳬲�
	vector<bool> whether_over_error_in_x_vectors(filter_mask_rows,false);	//x�����и����Ƿ񳬳���Χ
	vector<bool> whether_over_error_in_y_vectors(filter_mask_cols,false);	//y�����и����Ƿ񳬳���Χ

	for(int i=0;i<filter_mask_rows;i++)
	{
		normalized_x_vectors_ptr=normalized_x_vectors.ptr<float>(i);
		normalized_y_vectors_ptr=normalized_y_vectors.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			if(abs(normalized_x_vectors_ptr[j]-element_w_aver_in_x_vectors[j])>element_standard_in_x_vectors[j]+gamma/filter_mask_cols)//��i��x�����е�j��Ԫ��������
			{
				whether_over_error_in_x_vectors[i]=true;
			}
			if(abs(normalized_y_vectors_ptr[j]-element_w_aver_in_y_vectors[i])>element_standard_in_y_vectors[i]+gamma/filter_mask_rows)//��j��y�����е�i��Ԫ��������
			{
				whether_over_error_in_y_vectors[j]=true;
			}
		}
	}

	//ͳ�Ƴ�����С�������
	vector<int> over_error_y_in_x_vectors;	//x�����г�����Χ����
	vector<int> over_error_x_in_y_vectors;	//y�����г�����Χ����

	over_error_y_in_x_vectors.reserve(filter_mask_rows);
	over_error_x_in_y_vectors.reserve(filter_mask_cols);

	int over_error_rows=0;	//��������
	int over_error_cols=0;	//��������

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

	if(over_error_rows>filter_mask_rows-2||over_error_cols>filter_mask_cols-2)//����������й��࣬˵��û�зֽ�ı�Ҫ
	{
		flag_out=false;	//ģ���޷��ֽ�
		return true;
	}


	//6.���ó���Ǳ�����������������/����������ݼ���y,xģ������ĳ�������ֵ
	//	�����Ǳ�����������������/���������x,y��һ������֮��
	vector<float> primary_x_vector=element_sum_in_x_vectors;	//x������������
	vector<float> primary_y_vector=element_sum_in_y_vectors;	//y������������

	//�޳����������������Ԫ��ֵ
	for(int i=0;i<over_error_rows;i++)
	{
		normalized_x_vectors_ptr=normalized_x_vectors.ptr<float>(over_error_y_in_x_vectors[i]);

		for(int j=0;j<filter_mask_cols;j++)
		{
			primary_x_vector[j]-=normalized_x_vectors_ptr[j];
		}
	}
	//�޳����������������Ԫ��ֵ
	for(int i=0;i<filter_mask_rows;i++)
	{
		normalized_y_vectors_ptr=normalized_y_vectors.ptr<float>(i);

		for(int j=0;j<over_error_cols;j++)
		{
			primary_y_vector[i]-=normalized_y_vectors_ptr[over_error_x_in_y_vectors[j]];
		}
	}

	//	�õ���С�������ı���ϵ��min J=sum[(T(i,j)-alpha^2*Vx(i)*Vy(j))^2]
	float alpha_x=0;	//��С�������ı���ϵ��
	float alpha_y=0;

	float primary_x_vectors_part_model_square=0;	//x�����������ƵĲ���ģ��ƽ��(���������������)
	float primary_y_vectors_part_model_square=0;	//y�����������ƵĲ���ģ��ƽ��

	for(int i=0;i<filter_mask_cols;i++)//����x����ģ
	{
		if(whether_over_error_in_y_vectors[i]==false)//��������
		{
			primary_x_vectors_part_model_square+=primary_x_vector[i]*primary_x_vector[i];
		}
	}

	for(int i=0;i<filter_mask_rows;i++)//����y����ģ
	{
		if(whether_over_error_in_x_vectors[i]==false)//��������
		{
			primary_y_vectors_part_model_square+=primary_y_vector[i]*primary_y_vector[i];
		}
	}

	float x_y_vector_element_2_sum=primary_x_vectors_part_model_square*primary_y_vectors_part_model_square;	//sum(primary_x_vector^2*primary_y_vector^2)x,y����֮��Ԫ�ص�ƽ����

	for(int i=0;i<filter_mask_rows;i++)
	{
		if(whether_over_error_in_x_vectors[i])//����������
		{
			continue;
		}
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			if(whether_over_error_in_y_vectors[j])//����������
			{
				continue;
			}

			alpha_x+=filter_mask_ptr[j]*primary_y_vector[i]*primary_x_vector[j];//������С�������ı���ϵ��
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
	

	//	�õ�x,y��ģ�������������ֵ
	for(int i=0;i<filter_mask_cols;i++)//����x����ģ
	{
		primary_x_vector[i]=primary_x_vector[i]*alpha_x;
	}

	for(int i=0;i<filter_mask_rows;i++)//����y����ģ
	{
		primary_y_vector[i]=primary_y_vector[i]*alpha_y;
	}

	//	����ģ��Ԫ�ر�׼��ĳ�������ֵ
	standard_error=0;	//��׼��

	for(int i=0;i<filter_mask_rows;i++)
	{
		if(whether_over_error_in_x_vectors[i])//����������
		{
			continue;
		}
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			if(whether_over_error_in_y_vectors[j])//����������
			{
				continue;
			}

			float current_error=filter_mask_ptr[j]-primary_y_vector[i]*primary_x_vector[j];//���
			standard_error+=current_error*current_error;//����
		}
	}
	standard_error=sqrt(standard_error/(filter_mask_rows-over_error_rows)/(filter_mask_cols-over_error_cols));


	//7.��������x,yģ�壬�����ݱ�׼���Ǳ�ڷ��������ɸѡ����ʵ�������

	//���㳬���б����
	float over_error_threshold=gamma*sqrt(sum_element_sqaure/(filter_mask_rows*filter_mask_cols));//������ֵ=�����*ƽ������

	//������ֵ
	filter_mask_x=primary_x_vector;
	filter_mask_y=primary_y_vector;

	for(int itetate=0;itetate<5;itetate++)
	{
		//	��������x,yģ���������ƽ���ľ������ɸѡ����ǰ���������
		vector<cv::Vec2i> divorece_point;
		divorece_point.reserve(over_error_rows*over_error_cols);

		for(int i=0;i<filter_mask_rows;i++)
		{
			filter_mask_ptr=filter_mask.ptr<float>(i);

			for(int j=0;j<filter_mask_cols;j++)
			{
				float current_error=abs(filter_mask_ptr[j]-filter_mask_y[i]*filter_mask_x[j]);//���

				if(current_error>over_error_threshold)
				{
					divorece_point.push_back(cv::Vec2i(i,j));
				}
			}
		}
		
		//	��������x,y��ģ��������Ƶ�ǰ���������İ������ʣ��ֵ
		cv::Mat filter_mask_remains;	//���������ʣ��ֵ
		filter_mask.copyTo(filter_mask_remains);

		int divorece_point_number=divorece_point.size();
		for(int i=0;i<divorece_point_number;i++)//���������ֵ�޸�Ϊ����ֵ
		{
			int row=divorece_point[i][0];
			int col=divorece_point[i][1];

			filter_mask_remains.at<float>(row,col)=filter_mask_x[col]*filter_mask_y[row];
		}

		//	����ģ��Ļ���x,y��ֽ��㷨����x,yģ�����
		bool analyse_reult=basicFilterMaskXYAnalysis(filter_mask_remains,filter_mask_x,filter_mask_y,standard_error,flag_out);

		if(analyse_reult==false)	//���򱨴�
		{
			cout<<"error: in complexFilterMaskXYAnalysis"<<endl;
			return false;
		}

		if(flag_out==false)		//�޷��ֽ⣬���ͱ�׼��
		{
			over_error_threshold=0.8*over_error_threshold;
		}
		else
		{
			break;
		}
	}

	//8.�����������
	if(flag_out==false)	//������ɷ���
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
			float current_error=filter_mask_ptr[j]-filter_mask_y[i]*filter_mask_x[j];//���

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
bool basicFilterMaskXYconstAnalysis(cv::Mat& filter_mask,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& mask_const,float& standard_error,bool& flag_out)
{
	//0.����һ��xy�ֽ�
	vector<float> filter_mask_x_z,filter_mask_y_z;
	basicFilterMaskXYAnalysis(filter_mask,filter_mask_x_z,filter_mask_y_z,standard_error,flag_out);
	if(flag_out==true)
	{
		filter_mask_x=filter_mask_x_z;
		filter_mask_y=filter_mask_y_z;
		mask_const=0;
		return true;
	}

	//1.��ȡfilter_mask�ߴ�,���������ֵ��С
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	//add1(20170222)�������������ֵ������ƫ����
	float add_number=1.0/(filter_mask_rows*filter_mask_cols);	//��������ֵ��ƫ����
	float filter_st_mean=0;	//ģ���ʼ��ֵ
	float* filter_mask_ptr;					//ģ�����ָ��

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

	//2.����ģ���ֵ�����
	float filter_element_average=0;			//ģ��Ԫ�صľ�ֵ
	float filter_element_square_average=0;	//ģ��Ԫ�ص�ƽ����ֵ

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
	

	//3.ģ��0��ֵ��
	cv::Mat filter_mask_zero_aver;	//0��ֵ��ģ��
	filter_mask.convertTo(filter_mask_zero_aver,filter_mask.type());

	filter_mask_zero_aver=filter_mask_zero_aver-filter_element_average;


	//4.�ж�ģ������Ԫ���Ƿ���ͬ
	standard_error=0;
	bool same_flag=true;

	float* filter_mask_zero_aver_ptr;			//��ֵ��ģ�����ָ��
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
	if(same_flag==true)	//Ԫ��ȫ��ͬ�������ֵ
	{
		standard_error=sqrt(standard_error/(filter_mask_rows*filter_mask_cols));
		mask_const=filter_element_average;	//����Ϊģ���ֵ
		filter_mask_x.clear();				//x���������
		filter_mask_y.clear();				//y���������

		flag_out=true;
		return true;
	}


	//5.���û����ֽ����x,y�������µĹ���ֵ
	bool analyseed=false;
	if(abs(mask_const)<filter_element_square_average)	//����������ֵ����������ֵ����x,y����
	{
		cv::Mat filter_mask_2=filter_mask-mask_const;
		vector<float> filter_mask_x_z,filter_mask_y_z;
		analyseed=basicFilterMaskXYAnalysis(filter_mask_2,filter_mask_x_z,filter_mask_y_z,standard_error,flag_out);//ģ��Ļ���x,y��ֽ�
		if(analyseed==true)
		{
			filter_mask_x=filter_mask_x_z;
			filter_mask_y=filter_mask_y_z;
		}
	}
	else
	{
		vector<float> filter_mask_x_z,filter_mask_y_z;
		analyseed=basicFilterMaskXYAnalysis(filter_mask,filter_mask_x_z,filter_mask_y_z,standard_error,flag_out);//ģ��Ļ���x,y��ֽ�
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

	//6.�������xy����
	//����ֵת��Ϊ����
	cv::Mat f_m_x(filter_mask_x,CV_32FC1);
	cv::Mat f_m_y(filter_mask_y,CV_32FC1);

	//��������xy����
	for(int iterate=0;iterate<4*(filter_mask_cols+filter_mask_rows);iterate++)
	{
		float rate=sqrt(standard_error/filter_element_square_average);

		//(1) ����hx=[hy'hy*I-aver(hy)^2*E]^-1*[M'-aver(M)*E]*hy
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

		//(2) ����hy=[hx'hx*I-aver(hx)^2*E]^-1*[M-aver(M)*E]*hx
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

		//(3)�������������С����ֵ
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


	//7.����x������y�������㳣����
	float x_vector_average=0;	//x����Ԫ�ؾ�ֵ
	for(int i=0;i<filter_mask_cols;i++)
	{
		x_vector_average+=filter_mask_x[i];
	}
	x_vector_average=x_vector_average/filter_mask_cols;

	float y_vector_average=0;	//y����Ԫ�ؾ�ֵ
	for(int i=0;i<filter_mask_rows;i++)
	{
		y_vector_average+=filter_mask_y[i];
	}
	y_vector_average=y_vector_average/filter_mask_rows;

	mask_const=filter_element_average-x_vector_average*y_vector_average;


	//add_2(20170222)���ӳ��������޳�ƫ��������ԭƫ����
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


	//8.����������
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
bool complexFilterMaskXYconstAnalysis(cv::Mat& filter_mask,float& gamma,vector<float>& filter_mask_x,vector<float>& filter_mask_y,float& mask_const,vector<Point_Value>& delta_mask,float& standard_error,bool& flag_out)
{
	//0.��ȡfilter_mask�ߴ�,���������ֵ��С
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	//�ȳ���һ��ֱ�ӷֽ�
	basicFilterMaskXYconstAnalysis(filter_mask,filter_mask_x,filter_mask_y,mask_const,standard_error,flag_out);
	if(flag_out==true)
	{
		delta_mask.clear();
		return true;
	}

	//1.������С���֮��Ĳ����������һ��(�������Ź�һ��)
	//����������
	cv::Mat filter_mask_d_row=cv::Mat::zeros(filter_mask_rows-1,filter_mask_cols,filter_mask.type());	//�м���
	cv::Mat filter_mask_d_col=cv::Mat::zeros(filter_mask_rows,filter_mask_cols-1,filter_mask.type());	//�м���

	vector<int> d_row_sign(filter_mask_rows-1,0);	//���в�ֵ�Ԫ�ط��ź�
	vector<int> d_col_sign(filter_mask_cols-1,0);	//���в�ֵ�Ԫ�ط��ź�

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

	//������һ��
	cv::Mat filter_mask_drow_one=cv::Mat::zeros(filter_mask_rows-1,filter_mask_cols,filter_mask.type());	//��һ���м���
	cv::Mat filter_mask_dcol_one=cv::Mat::zeros(filter_mask_rows,filter_mask_cols-1,filter_mask.type());	//��һ���м���

	float* filter_mask_drow_one_ptr,*filter_mask_dcol_one_ptr;

	//(1)��һ��dx
	//����dx
	vector<int> sign_element_x(filter_mask_cols,0);	//ÿ��Ԫ�ط��ŷ���
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
	//У��dx
	vector<int> each_line_s_no_matched_n(filter_mask_rows-1,0);//���Ų�ƥ�����
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

	//(2)��һ��dy
	vector<int> sign_element_y(filter_mask_rows,0);	//ÿ��Ԫ�ط��ŷ���
	vector<float> dcol_model(filter_mask_cols,0);	//dy����ģ

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
	//У��dy
	vector<int> each_col_s_no_matched_n(filter_mask_cols-1,0);//���Ų�ƥ�����
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
	

	//2.�������й�һ���С�����������Ԫ�ؾ�ֵ������
	vector<float> aver_d_row(filter_mask_cols,0);
	vector<float> std_d_row(filter_mask_cols,0);
	vector<float> aver_d_col(filter_mask_rows,0);
	vector<float> std_d_col(filter_mask_rows,0);

	//��������ֵ
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

	//����������
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

	//dy��ֵ
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

	//dy����
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


	//3.�Է����ΪȨֵ�����Ȩ��ֵ����
	vector<float> aver_w_d_row(filter_mask_cols,0);
	vector<float> aver_w_d_col(filter_mask_rows,0);

	//�м�Ȩ��ֵ
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

	//dy��Ȩ��ֵ
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


	//4.�����������Ȩ��ֵ����֮�������÷������������
	vector<bool> divide_x(filter_mask_cols,false);	//��Ƿ�����
	vector<bool> divide_y(filter_mask_rows,false);	//��Ƿ�����

	vector<float> x_threshold(filter_mask_cols,0);	//x��ֵ
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

	vector<float> y_threshold(filter_mask_rows,0);	//y��ֵ
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

	//ͳ����ʹ�õ�������
	float x_used_number=0;	//ʹ�õ�����
	float y_used_number=0;	//ʹ�õ�����

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

	if(x_used_number==0||y_used_number==0||filter_mask_cols/3-x_used_number+filter_mask_rows/3-y_used_number>0&&x_used_number<2&&y_used_number<2)	//�������������Ŀ���࣬�÷���ﲻ�����ͼ�����������
	{
		flag_out=false;
		return true;
	}


	//5.����x,yģ�����������������Ԫ�س�ʼ����ֵ
	//�������ʽ
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

	//�������ϵ��
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


	//6.���㳣�����ʼ����ֵ
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


	//7.���ݷǷ������У�����x,yģ�������ĳ�ʼ����
	x_mask_prime=vector<float>(filter_mask_cols,0);
	y_mask_prime=vector<float>(filter_mask_rows,0);

	//ͳ�Ƹ�����������
	vector<int> row_sign(filter_mask_rows);	//����������
	vector<int> col_sign(filter_mask_cols);	//����������

	cv::Mat filter_mask_d=filter_mask-mask_const;//ģ������ȥ������
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

	//�����ʼ����
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

	//����ֵ������һ��
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


	//9.��������ģ��ֽ�
	filter_mask_x=x_mask_prime;
	filter_mask_y=y_mask_prime;

	//���������ֵ
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
		//(1)����hx,hy,const�������Ԫ�ص�dM,��������ֵ�жϷ�����
		vector<cv::Vec2i> divorece_point;
		divorece_point.reserve((filter_mask_cols-x_used_number)*(filter_mask_rows-y_used_number));

		for(int i=0;i<filter_mask_rows;i++)
		{
			filter_mask_ptr=filter_mask.ptr<float>(i);

			for(int j=0;j<filter_mask_cols;j++)
			{
				float current_error=abs(filter_mask_ptr[j]-filter_mask_y[i]*filter_mask_x[j]-mask_const);//���

				if(current_error>over_error_threshold)
				{
					divorece_point.push_back(cv::Vec2i(i,j));
				}
			}
		}

		//(2)���ڷ��������M(i,j)=hx(i)*hy(j)+const����ԭֵ
		cv::Mat filter_mask_remains;	//���������ʣ��ֵ
		filter_mask.copyTo(filter_mask_remains);

		int divorece_point_number=divorece_point.size();
		for(int i=0;i<divorece_point_number;i++)//���������ֵ�޸�Ϊ����ֵ
		{
			int row=divorece_point[i][0];
			int col=divorece_point[i][1];

			filter_mask_remains.at<float>(row,col)=filter_mask_x[col]*filter_mask_y[row]+mask_const;
		}

		//(3)���û���hx,hy,const�ֽ���Ʒֽ�ֵ�������׼��
		bool basic_result=basicFilterMaskXYconstAnalysis(filter_mask_remains,filter_mask_x,filter_mask_y,mask_const,standard_error,flag_out);

		if(basic_result==false)	//���򱨴�
		{
			cout<<"error: in complexFilterMaskXYconstAnalysis"<<endl;
		}

		if(flag_out==false)		//�޷��ֽ⣬���ͱ�׼��
		{
			over_error_threshold=0.9*over_error_threshold;
		}
		else
		{
			break;
		}
	}


	//10.�����������dM
	if(flag_out==false)	//������ɷ���
	{
		return true;
	}

	delta_mask.clear();
	delta_mask.reserve((filter_mask_cols-x_used_number)*(filter_mask_rows-y_used_number));

	bool vec_flag=true;	//���VEC���Ƿ���ڣ����������Ƿ�ֻ�г����
	if(filter_mask_y.size()==0||filter_mask_x.size()==0)
		vec_flag=false;

	for(int i=0;i<filter_mask_rows;i++)
	{
		filter_mask_ptr=filter_mask.ptr<float>(i);

		for(int j=0;j<filter_mask_cols;j++)
		{
			float current_error=filter_mask_ptr[j]-mask_const;//���

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

	if(delta_mask.size()>filter_mask_rows*filter_mask_cols/4)	//������������̫�࣬����ֽ�󲻻���������ٶ�
	{
		flag_out=false; 
	}

	return true;
}


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
bool vectorFilterMaskSymmetryAnalysis(vector<float>& filter_mask,float& gamma,vector<float>& symmetry_mask,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out)
{
	//1.��ȡfilter_mask�ߴ�,���������ֵ��С
	int filter_mask_size=filter_mask.size();	//��������
	int left_middle_position=filter_mask_size/2-1;	//�����м�ƫ�������

	symmetry_mask=vector<float>(filter_mask_size,0);	//ȷ����������ߴ�


	//2.���������ֵ
	float divide_threshold=0;
	for(int i=0;i<filter_mask_size;i++)
	{
		divide_threshold+=filter_mask[i]*filter_mask[i];
	}
	divide_threshold=gamma*sqrt(divide_threshold/(filter_mask_size*2));


	//3.��÷����
	delta_mask.clear();
	delta_mask.reserve(left_middle_position);
	vector<bool> is_delta_mask(filter_mask_size,false);	//��Ǹ����Ƿ�Ϊ������

	for(int i=0;i<left_middle_position+1;i++)
	{
		float d_mask_current=filter_mask[filter_mask_size-1-i]-filter_mask[i];
		if(abs(d_mask_current)>divide_threshold)
		{
			delta_mask.push_back(cv::Vec2f(filter_mask_size-1-i+0.001,d_mask_current));
			is_delta_mask[filter_mask_size-1-i]=true;
		}
	}
	if(delta_mask.size()>left_middle_position/2)	//����������࣬�÷���������
	{
		flag_out=false;
		return true;
	}

	//4.��������Գ�ģ��
	//�Ƿ���������ֵ
	for(int i=0;i<left_middle_position+1;i++)
	{
		float average=(filter_mask[i]+filter_mask[filter_mask_size-1-i])/2;	

		symmetry_mask[i]=average;
		symmetry_mask[filter_mask_size-1-i]=average;
	}
	if(2*left_middle_position+2<filter_mask_size)	//ģ����������,���Ƿ���ģ�帽���м���
	{
		symmetry_mask[left_middle_position+1]=filter_mask[left_middle_position+1];
	}

	//��������ǰ�벿�ֵ�ֵΪ׼
	int divided_number=delta_mask.size();
	for(int i=0;i<divided_number;i++)
	{
		int position=delta_mask[i][0];	//ע��˴�position>left_middle_position+1
		symmetry_mask[filter_mask_size-1-position]=filter_mask[filter_mask_size-1-position];
		symmetry_mask[position]=filter_mask[filter_mask_size-1-position];
	}

	//5.����������
	standard_error=0;

	for(int i=0;i<left_middle_position+1;i++)
	{
		if(is_delta_mask[filter_mask_size-1-i]==false)	//ֻ�зǷ�����������
		{
			float current_error=(filter_mask[filter_mask_size-1-i]-filter_mask[i]);//error=[T(j)-T(m-1-j)]/2,�������Գ�Ԫ��ͬ���
			standard_error+=current_error*current_error/2;
		}
	}
	standard_error=sqrt(standard_error/filter_mask_size);

	flag_out=true;
	return true;
}


/*	1.6 ���ƵȲ�����ģ��ķֽ�
	���룺
	cv::Mat& filter_mask			ģ������
	float& gamma					�����
	�����
	float& filter_constant			ģ���������ĳ�����
	float& filter_linear			ģ����������һ����
	vector<cv::Vec2f>& delta_mask	ϡ���������(x,df)
	float& standard_error
	bool flag_out					����ģ���Ƿ�ɲ�⣨tureΪ�ɲ�⣬��ʱģ���x,y���������Ч��
	����������falseʱ���������*/
bool gradeVectorFilterMaskAnalysisOriginal(vector<float>& filter_mask,float& gamma,float& filter_constant,float& filter_linear,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out)
{
	//1.��ȡfilter_mask�ߴ�,���������ֵ��С
	int filter_mask_size=filter_mask.size();	//��������


	//2.���������ֵ
	float divide_threshold=0;
	for(int i=0;i<filter_mask_size;i++)
	{
		divide_threshold+=filter_mask[i]*filter_mask[i];
	}
	divide_threshold=gamma*sqrt(divide_threshold/(filter_mask_size*2));


	//3.����ÿһ��dT(j)=T(j+1)-T(j),����÷����
	//����ÿһ��dT(j)=T(j+1)-T(j)
	vector<float> delta_filter_mask(filter_mask_size,0);

	for(int i=0;i<filter_mask_size-1;i++)
	{
		delta_filter_mask[i]=filter_mask[i+1]-filter_mask[i];
	}

	//����dT��ֵ�Լ���Ȩ��ֵ
	float delta_filter_mask_aver=(filter_mask[filter_mask_size-1]-filter_mask[0])/(filter_mask_size-1);//��ֵ
	float w_delta_filter_mask_aver=0;	//��Ȩ��ֵ
	float weight=0;	//Ȩ��

	for(int i=0;i<filter_mask_size-1;i++)
	{
		float d_delta_filter_mask=delta_filter_mask[i]-delta_filter_mask_aver;
		float d_delta_filter_mask_2=d_delta_filter_mask*d_delta_filter_mask;

		w_delta_filter_mask_aver+=delta_filter_mask[i]/(d_delta_filter_mask_2+0.001);
		weight+=1/(d_delta_filter_mask_2+0.001);
	}
	w_delta_filter_mask_aver=w_delta_filter_mask_aver/weight;

	//��÷����
	vector<float> d_delta_filter_mask(filter_mask_size-1,0);	//dT���Ȩ��ֵ֮��
	vector<bool> divide_filter_mask(filter_mask_size,true);		//���ÿ��Ԫ���Ƿ�Ϊ�����

	//����ÿdT�����
	for(int i=0;i<filter_mask_size-1;i++)
	{
		d_delta_filter_mask[i]=delta_filter_mask[i]-w_delta_filter_mask_aver;
	}

	//�����Ƿ�������ӵ� dT(j)==0&&dT(j+1)==0
	int first_seed=filter_mask_size;	//��ǵ�һ�����ӵ�����
	int last_seed=-1;					//������һ�����ӵ�����

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
	if(last_seed==-1)	//û�����ӵ㣬�޷��жϿɷ�����
	{
		flag_out=false;
		return true;
	}

	//���ݷǷ�������ӵ���Ʒ����
	for(int i=0;i<filter_mask_size;i++)
	{
		if(divide_filter_mask[i]==false)	//�����Ƿ������ӵ�
		{
			continue;
		}

		if(i<last_seed)//i֮�������ӵ㣬�����Ҫ�������
		{
			float sum_d_delta_filter=0;
			for(int j=i;j<last_seed;j++)
			{
				if(divide_filter_mask[j]==false)//��j��Ϊd(j+1)-d(j)
					break;

				sum_d_delta_filter+=d_delta_filter_mask[j];
			}
			if(abs(sum_d_delta_filter)<=divide_threshold)
			{
				divide_filter_mask[i]=false;
			}
		}
		else	//i֮��û�����ӵ㣬�����Ҫ��ǰ����
		{
			float sum_d_delta_filter=0;
			for(int j=i-1;j>=first_seed;j--)
			{
				sum_d_delta_filter+=d_delta_filter_mask[j];
				
				if(divide_filter_mask[j]==false)//��j��Ϊd(j+1)-d(j)
					break;
			}
			if(abs(sum_d_delta_filter)<=divide_threshold)
			{
				divide_filter_mask[i]=false;
			}
		}
	}

	//ͳ�Ʒ���������������m/2�ֽ������壬����
	int divide_number=0;	//�ֽ����
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

	//4.���㳣����a,һ����b�Ĺ���ֵ
	float sum_j=0;					//j�ĺ�
	float sum_j_square=0;			//j^2�ĺ�
	float sum_filter_mask=0;		//T(j)�ĺ�
	float w_sum_filter_mask=0;		//j*T(j)�ĺ�

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

	int no_divide_number=filter_mask_size-divide_number;	//�Ƿ��������

	filter_linear=(no_divide_number*w_sum_filter_mask-sum_j*sum_filter_mask)/(no_divide_number*sum_j_square-sum_j*sum_j);//һ����
	filter_constant=(sum_filter_mask-filter_linear*sum_j)/no_divide_number;//������
	
	if(abs(filter_linear/filter_constant)<0.01)	//һ����̫С�����Բ���
	{
		filter_constant=filter_constant*(0.5*filter_linear+1);
		filter_linear=0;
	}

	//5.����������Լ�������׼��
	delta_mask.clear();
	delta_mask.reserve(divide_number);

	standard_error=0;	//������׼��

	for(int i=0;i<filter_mask_size;i++)
	{
		float d_filter_mask_value=filter_mask[i]-filter_constant-i*filter_linear;//��ǰ�����
		if(divide_filter_mask[i]==true)	//�����
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
	//1.��ȡfilter_mask�ߴ�,���������ֵ��С
	int filter_mask_size=filter_mask.size();	//��������

	//2.���������ֵ
	float divide_threshold=0;
	for(int i=0;i<filter_mask_size;i++)
	{
		divide_threshold+=filter_mask[i]*filter_mask[i];
	}
	divide_threshold=gamma*sqrt(divide_threshold/filter_mask_size);

	//3.����newton���������������
	cv::Mat filter_para=cv::Mat::zeros(2,1,CV_32FC1);	//����������ֵ
	cv::Mat const_para(filter_mask,true);	//�������еĺ���������

	//������ֵ
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

	//��������
	bool lf_re=nolinearFunMinistNewtonSolver(sumSqrtFunc,divide_threshold*0.01,divide_threshold*0.01,filter_para,const_para);
	if(lf_re==false)
		return false;

	filter_constant=filter_para.at<float>(0);
	filter_linear=filter_para.at<float>(1);

	//������������
	vector<bool> divide_filter_mask(filter_mask_size,true);		//���ÿ��Ԫ���Ƿ�Ϊ�����

	for(int iter=0;iter<4;iter++)
	{
		//4.����������,Ԥ�з�����
		vector<bool> curr_divide_filter_mask(filter_mask_size,false);		//���ÿ��Ԫ���Ƿ�Ϊ�����
		int divide_number=0;	//�ֽ����

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

		//��������������m/2�ֽ������壬����
		if(divide_number>filter_mask_size/2)
		{
			flag_out=false;
			return true;
		}

		//6.��ǰ��������֮ǰ�Ա�,��ͬ������
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

		//5.�ų����������a,b����ֵ
		float sum_j=0;					//j�ĺ�
		float sum_j_square=0;			//j^2�ĺ�
		float sum_filter_mask=0;		//T(j)�ĺ�
		float w_sum_filter_mask=0;		//j*T(j)�ĺ�

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

		int no_divide_number=filter_mask_size-divide_number;	//�Ƿ��������

		filter_linear=(no_divide_number*w_sum_filter_mask-sum_j*sum_filter_mask)/(no_divide_number*sum_j_square-sum_j*sum_j);//һ����
		filter_constant=(sum_filter_mask-filter_linear*sum_j)/no_divide_number;//������

		if(abs(filter_linear*filter_mask_size/filter_constant)<0.01)	//һ����̫С�����Բ���
		{
			filter_constant+=filter_linear*(filter_mask_size-1)/2;
			filter_linear=0;
		}
	}
	//6.����������Լ�������׼��
	standard_error=0;	//������׼��
	int divide_number=0;
	delta_mask.clear();

	for(int i=0;i<filter_mask_size;i++)
	{
		float d_filter_mask_value=filter_mask[i]-filter_constant-i*filter_linear;//��ǰ�����
		if(divide_filter_mask[i]==true)	//�����
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
/*	����⺯�� min J=sum(sqrt(|h(j)-a-b*j|))
	��������	a=fun_in[0]
				b=fun_in[1]*/
void sumSqrtFunc(cv::Mat& base_para,cv::Mat& fun_in,float& fun_out)
{
	int n=max(base_para.rows,base_para.cols);	//�������ݸ���
	float* p_base_pare=(float*)base_para.data;
	float* p_fun_in=(float*)fun_in.data;

	fun_out=0;
	for(int j=0;j<n;j++)
		fun_out+=sqrt(abs(p_base_pare[j]-p_fun_in[0]-p_fun_in[1]*j));

}

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
bool parabolaVectorFilterMaskAnalysis(vector<float>& filter_mask,float& gamma,float& filter_constant,float& filter_linear,float& filter_p2,vector<cv::Vec2f>& delta_mask,float& standard_error,bool& flag_out)
{
	//1.��ȡfilter_mask�ߴ�,���������ֵ��С
	int filter_mask_size=filter_mask.size();	//��������

	//����׼��
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


	//2.���������ֵ
	float divide_threshold=0;
	for(int i=0;i<filter_mask_size;i++)
	{
		divide_threshold+=filter_mask[i]*filter_mask[i];
	}
	divide_threshold=gamma*sqrt(divide_threshold/filter_mask_size);

	//3.����newton���������������
	cv::Mat filter_para=cv::Mat::zeros(3,1,CV_32FC1);	//����������ֵ
	cv::Mat const_para(filter_mask,true);	//�������еĺ���������

	//������ֵ
	int half_size=filter_mask_size/2;
	int hhalf_size=half_size/2;

	filter_p2=0;		//2����
	for(int i=2*hhalf_size;i<filter_mask_size;i++)
		filter_p2+=filter_mask[i]+filter_mask[i-2*hhalf_size]-2*filter_mask[i-hhalf_size];

	filter_p2=filter_p2/(2*hhalf_size*hhalf_size*(filter_mask_size-2*hhalf_size));

	filter_linear=0;	//1����
	for(int i=half_size;i<filter_mask_size;i++)
		filter_linear+=filter_mask[i]-filter_mask[i-half_size];

	filter_linear=filter_linear/(half_size*(filter_mask_size-half_size))-filter_p2*filter_mask_size;

	filter_constant=0;	//0����
	for(int i=0;i<filter_mask_size;i++)
		filter_constant+=filter_mask[i];

	filter_constant=filter_constant/filter_mask_size-filter_linear*(filter_mask_size-1)/2-filter_p2*(filter_mask_size-1)*(filter_mask_size-1)/3;
	
	filter_para.at<float>(0)=filter_constant;
	filter_para.at<float>(1)=filter_linear;
	filter_para.at<float>(2)=filter_p2;

	//��������
	bool lf_re=nolinearFunMinistNewtonSolver(sumSqrtFunc2,divide_threshold*0.01,divide_threshold*0.01,filter_para,const_para);
	if(lf_re==false)
		return false;

	filter_constant=filter_para.at<float>(0);
	filter_linear=filter_para.at<float>(1);
	filter_p2=filter_para.at<float>(2);

	//������������
	vector<bool> divide_filter_mask(filter_mask_size,true);		//���ÿ��Ԫ���Ƿ�Ϊ�����

	for(int iter=0;iter<4;iter++)
	{
		//4.����������,Ԥ�з�����
		vector<bool> curr_divide_filter_mask(filter_mask_size,false);		//���ÿ��Ԫ���Ƿ�Ϊ�����
		int divide_number=0;	//�ֽ����
		standard_error=0;		//������׼��

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

		//��������������m/2�ֽ������壬����
		if(divide_number>filter_mask_size/2)
		{
			flag_out=false;
			return true;
		}


		//5.��ǰ��������֮ǰ�Ա�,��ͬ������
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


		//6.�ų����������ϵ������ֵ
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

		cv::solve(ATAuse,ATYuse,filter_para);	//���ϵ��

		filter_constant=filter_para.at<float>(0);
		filter_linear=filter_para.at<float>(1);
		filter_p2=filter_para.at<float>(2);

		if(abs(filter_p2*half_size*half_size/filter_constant)<0.01)	//2����̫С�����Բ���
		{
			filter_constant+=filter_p2*(2*filter_mask_size-1)*(filter_mask_size-1)/6;
			filter_p2=0;
		}

		if(abs(filter_linear*half_size/filter_constant)<0.01)	//1����̫С�����Բ���
		{
			filter_constant+=filter_linear*(filter_mask_size-1)/2;
			filter_linear=0;
		}
	}

	//6.����������Լ�������׼��
	standard_error=0;	//������׼��

	int divide_number=0;
	delta_mask.clear();
	for(int i=0;i<filter_mask_size;i++)
	{
		float d_filter_mask_value=filter_mask[i]-filter_constant-i*filter_linear-filter_p2*i*i;//��ǰ�����
		if(divide_filter_mask[i]==true)	//�����
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
	int n=max(base_para.rows,base_para.cols);	//�������ݸ���
	float* p_base_para=(float*)base_para.data;
	float* p_fun_in=(float*)fun_in.data;

	fun_out=0;
	for(int j=0;j<n;j++)
		fun_out+=sqrt(abs(p_base_para[j]-p_fun_in[0]-p_fun_in[1]*j-p_fun_in[2]*j*j));
}

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
bool integerMask(float gamma,cv::Mat& filter_mask,bool& flag_out,Vector_Mask& filter_mask_x,Vector_Mask& filter_mask_y,float& mask_const,vector<Point_Value>& xy_delta_mask,float& standard_error,Int_Mask_Divide& int_mask)
{
	//0.����ģ������
	int_mask.state=flag_out;	//���ģ��ɷ���
	int_mask.size=cv::Vec2i(filter_mask.rows,filter_mask.cols);

	if(flag_out==false)
	{
		computeRationalMatrix(gamma,filter_mask,int_mask.int_mask,int_mask.mask_denominator);
	}

	//1.x��������
	if(flag_out==true)
	{
		int_mask.vector_x.size=filter_mask_x.size;

		if(filter_mask_x.state==0)
		{
			computeRationalVector(gamma,filter_mask_x.basic_vector,int_mask.vector_x.basic_vector,int_mask.vector_x.denominator);
			int_mask.vector_x.state=0;
		}
		else if(filter_mask_x.state==1)	//�Գ�����
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

		if(filter_mask_x.state>0)//����������
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


	//2.y��������
	if(flag_out==true)
	{
		int_mask.vector_y.size=filter_mask_y.size;

		if(filter_mask_y.state==0)
		{
			computeRationalVector(gamma,filter_mask_y.basic_vector,int_mask.vector_y.basic_vector,int_mask.vector_y.denominator);
			int_mask.vector_y.state=0;
		}
		else if(filter_mask_y.state==1)	//�Գ�����
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

		if(filter_mask_y.state>0)//����������
		{
			int delta_size=filter_mask_y.delta_mask.size();
			int_mask.vector_y.delta_mask.resize(delta_size);

			for(int i=0;i<delta_size;i++)
			{
				int_mask.vector_y.delta_mask[i]=cv::Vec2i(filter_mask_y.delta_mask[i][0],filter_mask_y.delta_mask[i][1]*int_mask.vector_y.denominator+0.5);
			}
		}
	}


	//3.����������(����շת�����)
	realNumRational(gamma,mask_const,int_mask.mask_const[0],int_mask.mask_const[1]);


	//4.����������
	int xy_delta_mask_size=xy_delta_mask.size();
	int_mask.xy_delta_mask.resize(xy_delta_mask_size);

	for(int i=0;i<xy_delta_mask_size;i++)
	{
		int_mask.xy_delta_mask[i][0]=xy_delta_mask[i].col;
		int_mask.xy_delta_mask[i][1]=xy_delta_mask[i].row;
		realNumRational(gamma,xy_delta_mask[i].value,int_mask.xy_delta_mask[i][2],int_mask.xy_delta_mask[i][3]);
	}


	//5.�����׼��
	int filter_mask_rows=filter_mask.rows;
	int filter_mask_cols=filter_mask.cols;

	if(flag_out==false)	//ģ�岻�ɲ��
	{
		standard_error=standard_error*standard_error*filter_mask_rows*filter_mask_cols;//չ����׼��

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
		standard_error=standard_error*standard_error*filter_mask_rows*filter_mask_cols;//չ����׼��
		float x_add_error=0;
		float y_add_error=0;

		//x����
		vector<float> x_error(filter_mask_x.basic_vector.size(),0);

		if(filter_mask_x.state==0)
		{
			int vec_size=filter_mask_x.basic_vector.size();
			for(int i=0;i<vec_size;i++)
			{
				x_error[i]=filter_mask_x.basic_vector[i]-float(int_mask.vector_x.basic_vector[i])/int_mask.vector_x.denominator;
			}
		}

		if(filter_mask_x.state==1)	//�Գ�����
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
		int_mask.vector_x.standard_error=sqrt(x_add_error/int_mask.vector_x.size);	//x������������

		//y����
		vector<float> y_error(filter_mask_y.basic_vector.size(),0);

		if(filter_mask_y.state==0)
		{
			int vec_size=filter_mask_y.basic_vector.size();
			for(int i=0;i<vec_size;i++)
			{
				y_error[i]=filter_mask_y.basic_vector[i]-float(int_mask.vector_y.basic_vector[i])/int_mask.vector_y.denominator;
			}
		}

		if(filter_mask_y.state==1)	//�Գ�����
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
		int_mask.vector_y.standard_error=sqrt(y_add_error/int_mask.vector_y.size);	//y������������

		standard_error+=x_add_error*y_add_error;

		standard_error=sqrt(standard_error/(filter_mask_rows*filter_mask_cols));
	}

	//6.����������������
	int_mask.standard_error=standard_error;

	return true;
}


/*	ģ�����������Ժ���*/
bool filterMaskTest()
{
	//1 ģ��ֽ�
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
	bool analyseed=basicFilterMaskXYAnalysis(filter_mask,filter_mask_x,filter_mask_y,standard_error, flag_out);//ģ��Ļ���x,y��ֽ�

	if(analyseed==false)
		return false;

	vector<Point_Value> delta_mask;
	float gamma=0.2;	//�����
	analyseed=complexFilterMaskXYAnalysis(filter_mask,gamma,filter_mask_x,filter_mask_y,delta_mask,standard_error,flag_out);//����ģ���x,y��ֽ�

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
	cout<<"ģ�������߷ֽ���ʱ:"<<t_use0<<"s"<<endl;

	Vector_Mask new_filter_mask_x,new_filter_mask_y;
	vector<Point_Value> xy_delta_mask;
	analyseed=filterMaskAnalysis(filter_mask,gamma,new_filter_mask_x,new_filter_mask_y,xy_delta_mask,standard_error,flag_out);

	//ģ��x,y,const�ֽ�
	float mask_const;
	analyseed=basicFilterMaskXYconstAnalysis(filter_mask,filter_mask_x,filter_mask_y, mask_const,standard_error,flag_out);

	//complexFilterMaskXYconstAnalysis(filter_mask,gamma,filter_mask_x,filter_mask_y, mask_const,delta_mask,standard_error,flag_out);

	double t_start=(double)cv::getTickCount();
	analyseed=filterMaskAnalysis(filter_mask, gamma,new_filter_mask_x,new_filter_mask_y, mask_const, xy_delta_mask, standard_error, flag_out);
	double t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();;
	cout<<"ģ��ֽ���ʱ:"<<t_used<<"s"<<endl;

	//ģ�����λ�
	t_start=(double)cv::getTickCount();
	Int_Mask_Divide int_mask;
	integerMask(gamma,filter_mask,flag_out,new_filter_mask_x,new_filter_mask_y,mask_const,xy_delta_mask,standard_error,int_mask);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();;
	cout<<"ģ�����λ���ʱ:"<<t_used<<"s"<<endl;

	cout<<"����ģ��������Գɹ���"<<endl;
	return true;
}