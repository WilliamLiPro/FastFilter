/***************************************************************
	>����������ģ�壩����
	>���ߣ���ά��
	>��ϵ��ʽ��williamli_pro@163.com
	>ʵ��ͼ������жԳ�ģ����ǶԳ�ģ������
	>����Ҫ�㣺
	>1.ģ�����
	>2.ͼ�����ģ����������

****************************************************************/

#include "FastImageMaskFilter.h"

template<typename T_p> struct DividePointM	//��ά�����
{
	vector<int> row;
	vector<int> col;
	vector<T_p> value;
	vector<T_p> denominator;
};

template<typename T_p> struct DividePointL	//��ά�����
{
	vector<int> id;
	vector<T_p> value;
};

template<typename T_p1,typename T_p2,typename T_p3> bool deltaMaskImageFiltering(cv::Mat& image_in,cv::Vec2i& mask_size,DividePointM<T_p3>& delta_mask,cv::Mat& image_out);
template<typename T_p1,typename T_p2,typename T_p3> bool constMaskImageFiltering(cv::Mat& image_in,cv::Vec2i mask_size,T_p3 mask_const,cv::Mat& image_out);
template<typename T_p1,typename T_p2> void rowConstFilter(const uchar* src, uchar* dst,int xk_size,int width,int ch);
template<typename T_p1,typename T_p2> void colConstFilter(const uchar** src, uchar* dst,int yk_size, int nrows, int width,int ch);
template<typename T_p1,typename T_p2,typename T_p3> bool vectorMaskImageFilter(cv::Mat& image_in,cv::Mat& image_out,cv::Vec2i mask_size,
							vector<T_p3>& x_filter_common,vector<T_p3>& x_filter_symmetry,
							DividePointL<T_p3>& div_px,
							T_p3 x_const,T_p3 x_lnr,T_p3 x_p2,T_p3 x_deno,int xf_s,
							vector<T_p3>& y_filter_common,vector<T_p3>& y_filter_symmetry,
							DividePointL<T_p3>& div_py,
							T_p3 y_const,T_p3 y_lnr,T_p3 y_p2,T_p3 y_deno,int yf_s);
template<typename T_p1,typename T_p2> void rowCommonFilter(const uchar* src, uchar* dst,const uchar* xkernal,int xk_size,int width,int ch);
template<typename T_p1,typename T_p2> void colCommonFilter(const uchar** src, uchar* dst,const uchar* ykernal,int yk_size, int nrows, int width,int ch);
template<typename T_p1,typename T_p2> void rowSymmetryFilter(const uchar* src, uchar* dst,const uchar* xkernal,int xk_size,int width,int ch);
template<typename T_p1,typename T_p2> void colSymmetryFilter(const uchar** src, uchar* dst,const uchar* ykernal,int yk_size, int nrows, int width,int ch);
template<typename T_p1,typename T_p2> void rowGradeFilter(const uchar* src, uchar* dst,T_p2 k_const,T_p2 k_lnr,int xk_size,int width,int ch);
template<typename T_p1,typename T_p2> void colGradeFilter(const uchar** src, uchar* dst,T_p2 k_const,T_p2 k_lnr,int yk_size, int nrows, int width,int ch);
template<typename T_p1,typename T_p2> void rowParabolaFilter(const uchar* src, uchar* dst,T_p2 k_const,T_p2 k_lnr,T_p2 k_p2,int xk_size,int width,int ch);
template<typename T_p1,typename T_p2> void colParabolaFilter(const uchar** src, uchar* dst,T_p2 k_const,T_p2 k_lnr,T_p2 k_p2,int yk_size, int nrows, int width,int ch);
template<typename T_p1,typename T_p2,typename T_p3> void rowDivideFilter(const uchar* src, uchar* dst, DividePointL<T_p3>& div_px,int width,int ch);
template<typename T_p1,typename T_p2,typename T_p3> void colDivideFilter(const uchar** src, uchar* dst, DividePointL<T_p3>& div_py, int nrows, int width,int ch);



//����ģ��ֽ�Ŀ����˲���
FastImageMaskFilter::FastImageMaskFilter()
{
	//������ʼ��
	flag_divide_=false;
	mask_const_=1;
	standard_error_=0;

	filter_mask_=cv::Mat::ones(1,1,CV_32FC1);
	int_mask_.int_mask=cv::Mat::ones(1,1,CV_32SC1);
}

//�����˲�����
bool FastImageMaskFilter::runFastImageMaskFilter(cv::Mat& image_in,cv::Mat& image_out,const string data_type,bool image_edge)
{
	//float t_start=clock();//test
	//1.��Ե����
	//(1)��ȡ�������������Ϣ
	int filter_mask_rows=filter_mask_.rows;
	int filter_mask_cols=filter_mask_.cols;
	int im_in_depth=image_in.depth();
	if(filter_mask_rows==0||filter_mask_cols==0)
	{
		std::cout<<"error��->in FastImageMaskFilter::runFastImageMaskFilter��������˲�ģ����Ч (�ߴ�Ϊ0)"<<endl
			<<"����ģ�壬����ú�����inputMastFilter(cv::Mat& filter_mask,float gamma)"<<endl;
		return false;
	}

	//(2)��չ����ͼ��
	cv::Mat image_in_used;
	bool flag=true;//=false;
	if(image_edge)
	{
		cv::copyMakeBorder(image_in,image_in_used,(filter_mask_rows-1)/2,(filter_mask_rows-1)/2,(filter_mask_cols-1)/2,(filter_mask_cols-1)/2,cv::BORDER_WRAP);
		//flag=imageExtension(image_in,cv::Vec2i(filter_mask_rows,filter_mask_cols),image_in_used);
	}
	else
		image_in_used=image_in;

	if(flag==false)
	{
		std::cout<<"error��->in FastImageMaskFilter::runFastImageMaskFilter��"<<endl;
		return false;
	}

	//(3)���������������ת������ͼ������
	if(data_type=="uchar"||data_type=="char"||data_type=="ushort"||data_type=="short"||data_type=="int"||image_in_used.depth()<CV_32S&&data_type=="")	//�����������
	{
		if(im_in_depth!=CV_32S)
			image_in_used.convertTo(image_in_used,CV_32S);
	}
	else if(data_type=="float"||image_in_used.depth()==CV_32F&&data_type=="")	//����������
	{
		if(im_in_depth!=CV_32F)
			image_in_used.convertTo(image_in_used,CV_32F);
	}
	else
	{
		cout<<"error: -> in FastImageMaskFilter::runFastImageMaskFilter :"<<endl
			<<"	      �������Ͳ��� data_type �������κ���֪��������"<<endl;
		return false;
	}
	//float t_used=(clock()-t_start)/1000.0;//test
	//cout<<"ǰ��׼������������ʱ: "<<t_used<<" s"<<endl;//test

	//t_start=clock();//test


	//2.����ģ��ֽ���ȷ���Ƿ�ִ��2D�˲�
	bool filter_flag=false;
	if(flag_divide_==false)
	{
		filter_flag=tDmaskImageFiltering(image_in_used,filter_mask_,data_type,image_out);
		if(filter_flag==false)
		{
			cout<<"error: -> in FastImageMaskFilter::runFastImageMaskFilter ->"<<endl;
			return false;
		}
		return true;
	}


	//3.����ģ��ֽ����ִ�зֽ��˲�
	cv::Mat mat_const,mat_delta,mat_xy;	//��˳��ֱ�Ϊ�������˲�������������˲������xy���˲����
	bool const_flag=false,delta_flag=false,xy_flag=false;	//�����˲��������Ч�Ա��

	//(2)���㳣�����˲�
	if(image_in_used.depth()==CV_32S&&int_mask_.mask_const[0]!=0)	//����������
	{
		filter_flag=constMaskImageFiltering<int,int>(image_in_used,int_mask_.size,int_mask_.mask_const[0],mat_const);
		if(filter_flag==false)
		{
			cout<<"error: -> in FastImageMaskFilter::runFastImageMaskFilter ->"<<endl;
			return false;
		}
		if(int_mask_.mask_const[1]!=1)
		{
			if(int_mask_.mask_const[1]==-1)
				mat_const=-mat_const;
			else
				mat_const=mat_const/int_mask_.mask_const[1];
		}
		const_flag=true;
	}
	else if(mask_const_!=0)	//����������
	{
		filter_flag=constMaskImageFiltering<float,float>(image_in_used,cv::Vec2i(filter_mask_.rows,filter_mask_.cols),mask_const_,mat_const);
		if(filter_flag==false)
		{
			cout<<"error: -> in FastImageMaskFilter::runFastImageMaskFilter ->"<<endl;
			return false;
		}
		const_flag=true;
	}

	//(3)����������˲�
	if(image_in_used.depth()==CV_32S&&int_mask_.xy_delta_mask.size()>0)	//����������
	{
		DividePointM<int> xy_delta_m;
		int delta_n=int_mask_.xy_delta_mask.size();
		
		xy_delta_m.row.resize(delta_n);
		xy_delta_m.col.resize(delta_n);
		xy_delta_m.value.resize(delta_n);
		xy_delta_m.denominator.resize(delta_n);

		for(int i=0;i<delta_n;i++)
		{
			xy_delta_m.row[i]=int_mask_.xy_delta_mask[i][1];
			xy_delta_m.col[i]=int_mask_.xy_delta_mask[i][0];
			xy_delta_m.value[i]=int_mask_.xy_delta_mask[i][2];
			xy_delta_m.denominator[i]=int_mask_.xy_delta_mask[i][3];
		}

		filter_flag=deltaMaskImageFiltering<int,int>(image_in_used,int_mask_.size,xy_delta_m,mat_delta);
		if(filter_flag==false)
		{
			cout<<"error: -> in FastImageMaskFilter::runFastImageMaskFilter ->"<<endl;
			return false;
		}
		delta_flag=true;
	}
	else if(xy_delta_mask_.size()>0)	//����������
	{
		DividePointM<float> xy_delta_m;
		int delta_n=xy_delta_mask_.size();
		
		xy_delta_m.row.resize(delta_n);
		xy_delta_m.col.resize(delta_n);
		xy_delta_m.value.resize(delta_n);
		xy_delta_m.denominator.resize(delta_n);

		for(int i=0;i<delta_n;i++)
		{
			xy_delta_m.row[i]=xy_delta_mask_[i].row;
			xy_delta_m.col[i]=xy_delta_mask_[i].col;
			xy_delta_m.value[i]=xy_delta_mask_[i].value;
			xy_delta_m.denominator[i]=1;
		}

		filter_flag=deltaMaskImageFiltering<float,float>(image_in_used,int_mask_.size,xy_delta_m,mat_delta);
		if(filter_flag==false)
		{
			cout<<"error: -> in FastImageMaskFilter::runFastImageMaskFilter ->"<<endl;
			return false;
		}
		delta_flag=true;
	}

	//(4)����xy�˲�
	if(filter_mask_x_.size>0&&filter_mask_y_.size>0)
	{
		if(image_in_used.depth()==CV_32S&&filter_mask_x_.size>0)	//����������
		{
			filter_flag=vectorMaskImageFiltering(image_in_used,filter_mask_x_,filter_mask_y_,int_mask_,"int",mat_xy);
			if(filter_flag==false)
			{
				cout<<"error: -> in FastImageMaskFilter::runFastImageMaskFilter ->"<<endl;
				return false;
			}
			xy_flag=true;
		}
		else	//����������
		{
			filter_flag=vectorMaskImageFiltering(image_in_used,filter_mask_x_,filter_mask_y_,int_mask_,"float",mat_xy);
			if(filter_flag==false)
			{
				cout<<"error: -> in FastImageMaskFilter::runFastImageMaskFilter ->"<<endl;
				return false;
			}
			xy_flag=true;
		}
	}
	//t_used=(clock()-t_start)/1000.0;//test
	//cout<<"�˲���ʱ: "<<t_used<<" s"<<endl;//test

	//t_start=clock();//test
	//3.�ϲ������˲����
	if(xy_flag)
	{
		if(const_flag)
		{
			image_out=mat_xy+mat_const;
		}
		else
		{
			image_out=mat_xy;
		}
		if(delta_flag)
		{
			image_out=image_out+mat_delta;
		}
	}
	else
	{
		if(const_flag&&delta_flag)
		{
			image_out=mat_const+mat_delta;
		}
		else if(const_flag)
		{
			image_out=mat_const;
		}
	}

	//4.���ͼ��ı�Ϊָ������
	if(data_type=="uchar")
	{
		if(image_out.depth()!=CV_8U)
			image_out.convertTo(image_out,CV_8U);
	}
	else if(data_type=="char")
	{
		if(image_out.depth()!=CV_8S)
			image_out.convertTo(image_out,CV_8S);
	}
	else if(data_type=="ushort")
	{
		if(image_out.depth()!=CV_16U)
			image_out.convertTo(image_out,CV_16U);
	}
	else if(data_type=="short")
	{
		if(image_out.depth()!=CV_16S)
		image_out.convertTo(image_out,CV_16S);
	}
	else if(data_type=="int")
	{
		if(image_out.depth()!=CV_32S)
		image_out.convertTo(image_out,CV_32S);
	}
	else if(data_type=="float")
	{
		if(image_out.depth()!=CV_32F)
		image_out.convertTo(image_out,CV_32F);
	}
	else if(data_type=="")	//������ͼ��������ͬ
	{
		if(image_out.depth()!=im_in_depth)
			image_out.convertTo(image_out,im_in_depth);
	}
	//t_used=(clock()-t_start)/1000.0;//test
	//cout<<"�����ʱ: "<<t_used<<" s"<<endl;//test

	return true;
}

//ģ�����뺯��
bool FastImageMaskFilter::inputMastFilter(cv::Mat& filter_mask,float gamma)
{
	//1.��׼������ģ������(32λfloat)
	int filter_mask_data_type=filter_mask.depth();	//��ȡģ����������� 
	if(filter_mask_data_type!=CV_32F)
	{
		filter_mask.convertTo(filter_mask_,CV_32F);
	}
	else
		filter_mask.copyTo(filter_mask_);

	//2.ִ��ģ�����
	int t_used=clock();
	//(1)��ģ���˲�
	mask_const_=0;
	bool analyse_result=filterMaskAnalysis(filter_mask_, gamma, filter_mask_x_, filter_mask_y_,mask_const_, xy_delta_mask_,standard_error_, flag_divide_);
	/*if(flag_divide_==false)
	{
		//(2)����ģ���˲�
		analyse_result=filterMaskAnalysis(filter_mask_, gamma, filter_mask_x_, filter_mask_y_,mask_const_, xy_delta_mask_,standard_error_, flag_divide_);
	}*/
	if(abs(mask_const_)<1E-10)
		mask_const_=0;

	time_used[0] = (clock()-t_used)/1000.0;
	std::cout<<"ģ�������ʱ: "<<time_used[0]<<" s"<<endl;

	if(analyse_result==false)
	{
		std::cout<<"error��-> in FastImageMaskFilter::inputMastFilter��"<<endl;
		return false;
	}
	if(flag_divide_==false)
	{
		std::cout<<"warning: ģ���޷��ֽ⣬����2Dģ����㣨���ܻή�������ٶȣ�"<<endl;
	}
	
	//3.ģ������
	t_used=clock();
	bool out_flag=integerMask(gamma,filter_mask_,flag_divide_,filter_mask_x_,filter_mask_y_,mask_const_,xy_delta_mask_,standard_error_,int_mask_);
	time_used[1] = (clock()-t_used)/1000.0;
	std::cout<<"ģ��������ʱ: "<<time_used[1]<<" s"<<endl;
	if(out_flag==false)
	{
		std::cout<<"error��-> in FastImageMaskFilter::inputMastFilter��"<<endl;
		return false;
	}

	return true;
}


/***************	�������Ӻ���	**************/


/*	����ģ������(�����ú���)	
	���룺
	cv::Mat& image_in			����ͼ��(��ͨ�����ͨ������������)
	cv::Mat& filter_mask		������ͼ�������������ģ��(��ͨ��,float�ͣ�
	float& gamma				ģ��ֽ��������ֵ
	int data_type				���ͼ���������ͣ�0:uchar��1:float��
	int mask_divide_type		ģ��ֽ����ͣ�0:���ֽ⣻1:xy�ֽ⣻2:xy�����ֽ⣩
	bool filters_result_expand	ģ���������Ƿ���չ��true����չ������������������ߴ���ͬ��������ģ���С�۳���Ӧ�ߴ磩
	�����
	float& standard_error		ģ��ֽ������׼��
	cv::Mat& image_out			���ͼ��(float��)

	����������falseʱ���������*/
//bool imageFiltering(cv::Mat& image_in_origine,cv::Mat& filter_mask,float& gamma,int data_type,int mask_divide_type,bool filters_result_expand,float& standard_error,cv::Mat& image_out)
//{
//	//0.��ȡ����ͼ���Լ�ģ���ģ
//	int image_in_rows,image_in_cols,filter_mask_rows,filter_mask_cols;
//
//	filter_mask_rows=filter_mask.rows;
//	filter_mask_cols=filter_mask.cols;
//
//	cv::Mat image_in;
//	if(filters_result_expand)
//	{
//		imageExtension(image_in_origine,cv::Vec2i(filter_mask_rows,filter_mask_cols),image_in);
//	}
//
//	//1.��������ͼ���������ͣ�ת����Ҫ������
//	int image_data_type=image_in.depth();		//��ȡ����ͼ�����������
//	int filter_mask_data_type=filter_mask.depth();	//��ȡģ����������� 
//	
//	//��׼������ͼ������
//	if(data_type==0)
//	{
//		if(image_data_type!=CV_32S)	//��׼������ͼ������(int)
//		{
//			image_in.convertTo(image_in,CV_32S);
//		}
//	}
//	if(data_type==1)
//	{
//		if(image_data_type!=CV_32F)	//��׼������ͼ������(32float)
//		{
//			image_in.convertTo(image_in,CV_32F);
//		}
//	}
//	
//	//��׼������ģ������(32λfloat)
//	if(filter_mask_data_type!=CV_32F)
//	{
//		filter_mask.convertTo(filter_mask,CV_32F);
//	}
//
//	//3.ִ��ģ�����
//	//(1)float��ģ�����
//	Vector_Mask filter_mask_x;
//	Vector_Mask filter_mask_y;
//	float mask_const=0;
//	bool flag_out=false;
//	vector<Point_Value> xy_delta_mask;
//
//	if(mask_divide_type==0)	//���ֽ�
//	{
//		flag_out=false;	//ģ�岻�ֽ�
//	}
//	if(mask_divide_type==1)	//xy�ֽ�
//	{
//		//float t_time=(double)cv::getTickCount();//test
//		bool analyse_result=filterMaskAnalysis(filter_mask, gamma, filter_mask_x, filter_mask_y, xy_delta_mask,standard_error, flag_out);
//		//t_time = ((double)cv::getTickCount()-t_time)/cv::getTickFrequency();
//		//cout<<"Analysis time used: "<<t_time<<" s"<<endl;
//
//		if(analyse_result==false)
//		{
//			cout<<"error��-> in fastImageFiltering��"<<endl;
//			return false;
//		}
//		if(flag_out==false)
//			cout<<"ģ���޷��ֽ⣬����2Dģ����㣨���ܻή�������ٶȣ�"<<endl;
//	}
//	if(mask_divide_type==2)	//xy�����ֽ�
//	{
//		//float t_time=(double)cv::getTickCount();//test
//		bool analyse_result=filterMaskAnalysis(filter_mask, gamma, filter_mask_x, filter_mask_y,mask_const,xy_delta_mask,standard_error, flag_out);
//		//t_time = ((double)cv::getTickCount()-t_time)/cv::getTickFrequency();
//		//cout<<"Analysis time used: "<<t_time<<" s"<<endl;
//
//		if(analyse_result==false)
//		{
//			cout<<"error��-> in fastImageFiltering��"<<endl;
//			return false;
//		}
//		if(flag_out==false)
//			cout<<"ģ���޷��ֽ⣬����2Dģ����㣨���ܻή�������ٶȣ�"<<endl;
//	}
//
//	//(2)ģ������
//	Int_Mask_Divide int_mask;
//	if(data_type==0)
//	{
//		if(mask_divide_type!=2)	//������������
//			mask_const=0;
//
//		bool out_flag=integerMask(gamma,filter_mask,flag_out,filter_mask_x,filter_mask_y,mask_const,xy_delta_mask,standard_error,int_mask);
//	}
//
//	//test
//	/*if(mask_divide_type==2&&data_type==0)
//	{
//		float filter_mask_x_l_error=1.0-filter_mask_x.filter_linear/(1.0*int_mask.vector_x.filter_linear/int_mask.vector_x.denominator);
//		float filter_mask_x_c_error=1.0-filter_mask_x.filter_constant/(1.0*int_mask.vector_x.filter_constant/int_mask.vector_x.denominator);
//		float filter_mask_y_l_error=1.0-filter_mask_y.filter_linear/(1.0*int_mask.vector_y.filter_linear/int_mask.vector_y.denominator);
//		float filter_mask_y_c_error=1.0-filter_mask_y.filter_constant/(1.0*int_mask.vector_y.filter_constant/int_mask.vector_y.denominator);
//
//		cout<<"������һ��"<<filter_mask_x_l_error<<","<<filter_mask_x_c_error<<","<<filter_mask_y_l_error<<","<<filter_mask_y_c_error<<","<<endl;
//	}*/
//
//
//	//4.��ͬ����µ�ģ���˲�
//	bool filter_flag=true;	//����˲�����
//
//	//(1)����2Dģ���˲�
//	if(mask_divide_type==0||mask_divide_type==1&&flag_out==false)	//ѡ�񲻷ֽ���߷ֽⲻ��
//	{
//		if(data_type==1)	//float
//			filter_flag=tDmaskImageFiltering(image_in,filter_mask,image_out);	
//		else	//uint8
//		{
//			filter_flag=tDmaskImageFiltering(image_in,int_mask,image_out);
//		}
//
//		if(filter_flag==false)
//		{
//			cout<<"error��-> in fastImageFiltering��"<<endl;
//			return false;
//		}
//
//		return true;
//	}
//
//	//(2)��ͬ�����µ��˲��������
//	bool have_delta_m=false;
//	cv::Mat const_filte_result;	//�����˲����
//	cv::Mat delta_filte_result;	//2D�������˲����
//	cv::Mat xy_filte_result;	//xy�˲����
//
//	if(mask_divide_type>0)	//����:2D�������˲�,xy�ֽ��˲�
//	{
//		//2D�������˲�
//		if(data_type==1)	//float
//		{
//			if(xy_delta_mask.size()>0)
//			{
//				cv::Vec2i mask_size=cv::Vec2i(filter_mask_rows,filter_mask_cols);
//				filter_flag=deltaMaskImageFiltering(image_in,mask_size,xy_delta_mask,delta_filte_result);	//float
//				have_delta_m=true;
//			}
//		}
//		else	//int
//		{
//			if(int_mask.xy_delta_mask.size()>0)
//			{
//				filter_flag=deltaMaskImageFiltering(image_in,int_mask,delta_filte_result);//int
//				have_delta_m=true;
//			}
//		}
//
//		if(filter_flag==false)
//		{
//			cout<<"error��-> in fastImageFiltering��"<<endl;
//			return false;
//		}
//
//		//xy�˲����
//		if(data_type==1&&filter_mask_x.size>0&&filter_mask_y.size>0)	//float
//		{
//			filter_flag=vectorMaskImageFiltering(image_in,filter_mask_x,filter_mask_y,xy_filte_result);	//float
//			if(filter_flag==false)
//			{
//				cout<<"error��-> in fastImageFiltering��"<<endl;
//				return false;
//			}
//		}
//		else	//int
//		{
//			if(int_mask.vector_x.size>0&&int_mask.vector_y.size>0)
//			{
//				filter_flag=vectorMaskImageFiltering( image_in,int_mask.vector_x, int_mask.vector_y,xy_filte_result);	//int
//				if(filter_flag==false)
//				{
//					cout<<"error��-> in fastImageFiltering��"<<endl;
//					return false;
//				}
//			}
//		}
//	}
//
//	if(mask_divide_type==2)	//����:�����˲����
//	{
//		//�����˲����
//		if(data_type==1&&mask_const!=0)	//float
//		{
//			//float t_time=(double)cv::getTickCount();//test
//			//filter_flag=constMaskImageFiltering( image_in, cv::Vec2i(filter_mask.rows,filter_mask.cols),mask_const,const_filte_result);	//float
//			//t_time = ((double)cv::getTickCount()-t_time)/cv::getTickFrequency();
//			//cout<<"Const Filte time used: "<<t_time<<" s"<<endl;
//
//			if(filter_flag==false)
//			{
//				cout<<"error��-> in fastImageFiltering��"<<endl;
//				return false;
//			}
//		}
//		else	//int
//		{
//			if(int_mask.mask_const[0]!=0)
//			{
//				filter_flag=constMaskImageFiltering( image_in,int_mask,const_filte_result);	//int
//				if(filter_flag==false)
//				{
//					cout<<"error��-> in fastImageFiltering��"<<endl;
//					return false;
//				}
//			}
//		}
//	}
//
//	//(3)�ϲ��˲����
//	if(mask_divide_type==1)	//xy�ֽ�
//	{
//		if(have_delta_m==true)	//����������
//			image_out=xy_filte_result+delta_filte_result;
//		else
//			xy_filte_result.copyTo(image_out);
//	}
//
//	if(mask_divide_type==2)	//xy�����ֽ�
//	{
//		if(data_type!=0&&mask_const!=0||data_type==0&&int_mask.mask_const[0]!=0)	//����������
//		{
//			if(have_delta_m==true)	//����������
//				image_out=const_filte_result+xy_filte_result+delta_filte_result;
//			else
//				image_out=const_filte_result+xy_filte_result;
//		}
//		else
//		{
//			if(have_delta_m==true)	//����������
//				image_out=xy_filte_result+delta_filte_result;
//			else
//				xy_filte_result.copyTo(image_out);
//		}
//
//		/*//test
//		if(data_type==0)
//		{
//			cv::imshow("int const_filte_result",const_filte_result*255);
//			cv::imshow("int xy_filte_result",xy_filte_result*255);
//			if(have_delta_m==true)
//				cv::imshow("int delta_filte_result",delta_filte_result*255);
//		}
//		else
//		{
//			cv::imshow("float const_filte_result",const_filte_result/255);
//			cv::imshow("float xy_filte_result",xy_filte_result/255);
//			if(have_delta_m==true)
//				cv::imshow("float delta_filte_result",delta_filte_result/255);
//		}
//		cv::waitKey(10000);*/
//	}
//
//	if(data_type==0)//8λuint������
//		image_out.convertTo(image_out,CV_8U);
//
//	return true;
//}

/*	ͼ����չ����
	���룺
	cv::Mat& image_in			����ͼ��(��ͨ�����ͨ������������)
	cv::Vec2i& mask_size		ģ��ߴ�		
	�����
	cv::Mat& image_out			���ͼ��*/
bool imageExtension(cv::Mat& image_in,cv::Vec2i& mask_size,cv::Mat& image_out)
{
	//1.��ȡ����ͼ���Լ�ģ���������
	int image_in_rows,image_in_cols;

	image_in_rows=image_in.rows;
	image_in_cols=image_in.cols;

	//�����м����
	cv::Mat image_middle;
	if(image_in.ptr<int>(0)==image_out.ptr<int>(0))
	{
		image_in.copyTo(image_middle);
	}
	else
		image_middle=image_in;

	//������������С
	int image_out_rows,image_out_cols;
	image_out_rows=image_in_rows+mask_size[0]-1;
	image_out_cols=image_in_cols+mask_size[1]-1;

	int c_x,c_y;
	c_x=(mask_size[1]-1)/2;
	c_y=(mask_size[0]-1)/2;

	//��������ͼ�������ͼ���еķ�Χ
	int max_x,max_y;
	max_x=c_x+image_in_cols-1;
	max_y=c_y+image_in_rows-1;

	//��ʼ���������
	image_out.create(image_out_rows,image_out_cols,image_in.type());

	//2.������չ����
	image_middle.copyTo(image_out(cv::Rect(c_x,c_y,image_in_cols,image_in_rows)));
	
	for(int i=0;i<c_y;i++)
	{
		image_middle(cv::Rect(0,0,image_in_cols,1)).copyTo(image_out(cv::Rect(c_x,i,image_in_cols,1)));
	}
	for(int i=max_y;i<image_out_rows;i++)
	{
		image_middle(cv::Rect(0,image_in_cols-1,image_in_cols,1)).copyTo(image_out(cv::Rect(c_x,i,image_in_cols,1)));
	}
	for(int i=0;i<c_x;i++)
	{
		image_out(cv::Rect(c_x,0,1,image_out_rows)).copyTo(image_out(cv::Rect(i,0,1,image_out_rows)));
	}
	for(int i=max_x;i<image_out_cols;i++)
	{
		image_out(cv::Rect(max_x-1,0,1,image_out_rows)).copyTo(image_out(cv::Rect(i,0,1,image_out_rows)));
	}
	
	return true;;
}

/*	2.1 2Dģ���˲�
	���룺
	cv::Mat& image_in			����ͼ��(��ͨ�����ͨ������������)
	cv::Mat& filter_mask		������ͼ�������������ģ��(��ͨ����
	const string data_type		���ͼ���������ͣ�uchar char short int float,Ĭ����image_in������ͬ��
	�����
	cv::Mat& image_out			���ͼ��*/
bool tDmaskImageFiltering(cv::Mat& image_in,cv::Mat& filter_mask,const string data_type,cv::Mat& image_out)
{
	//ֱ�ӵ���opencv����
	if(data_type=="uchar")
	{
		cv::filter2D(image_in,image_out,CV_8U,filter_mask);
		return true;
	}
	if(data_type=="char")
	{
		cv::filter2D(image_in,image_out,CV_8S,filter_mask);
		return true;
	}
	if(data_type=="ushort")
	{
		cv::filter2D(image_in,image_out,CV_16U,filter_mask);
		return true;
	}
	if(data_type=="short")
	{
		cv::filter2D(image_in,image_out,CV_16S,filter_mask);
		return true;
	}
	if(data_type=="int")
	{
		cv::filter2D(image_in,image_out,CV_32S,filter_mask);
		return true;
	}
	if(data_type=="float")
	{
		cv::filter2D(image_in,image_out,CV_32F,filter_mask);
		return true;
	}
	cv::filter2D(image_in,image_out,-1,filter_mask);
	return true;
}

/*	2.2 ��ά�������˲�	
	���룺
	cv::Mat& image_in				����ͼ��(��ͨ�����ͨ������������)
	cv::Vec2i& mask_size			ģ��ߴ�
	DividePointM& delta_mask		������
	�����
	cv::Mat& image_out				���ͼ��*/
template<typename T_p1,typename T_p2,typename T_p3> bool deltaMaskImageFiltering(cv::Mat& image_in,cv::Vec2i& mask_size,DividePointM<T_p3>& delta_mask,cv::Mat& image_out)
{
	//1.ͳ��������Ϣ
	int im_in_rows=image_in.rows,im_in_cols=image_in.cols,
		xf_size=mask_size[1],yf_size=mask_size[0];							//����ͼ�����˲���������
	int im_in_type=image_in.depth();										//����ͼ����������
	int d_m_size=delta_mask.row.size();
	int im_out_rows=im_in_rows-yf_size+1,im_out_cols=im_in_cols-xf_size+1;	//���ͼ��������
	int im_ch=image_in.channels();									//ͼ��ͨ����

	//2.�������ͼ��
	//T_p2 type_test=T_p2(1);
	int im_out_type=CV_32F;
	if(typeid(T_p2)==typeid(uchar))
		im_out_type=CV_8U;
	else if(typeid(T_p2)==typeid(ushort))
		im_out_type=CV_16U;
	else if(typeid(T_p2)==typeid(short))
		im_out_type=CV_16S;
	else if(typeid(T_p2)==typeid(int))
		im_out_type=CV_32S;
	else if(typeid(T_p2)==typeid(float))
		im_out_type=CV_32F;

	image_out.create(im_out_rows,im_out_cols,CV_MAKETYPE(im_out_type,im_ch));

	//3.����������ͼ�����ָ��
	const uchar** im_in_ptr=(const uchar**) malloc(im_in_rows*sizeof(uchar*));	//����ͼ��ָ��
	for(int i=0;i<im_in_rows;i++)
	{
		im_in_ptr[i]=image_in.ptr<uchar>(i);
	}

	//4.�����˲�
	int line_sz=im_out_cols*im_ch;
	int i,j,s;

	for( i=0;i<im_out_rows;i++,im_in_ptr++)//,im_out_ptr++)
	{
		T_p2* D = (T_p2*)image_out.data+i*line_sz;	//���е���

		int d_row=delta_mask.row[0];
		int d_col=delta_mask.col[0];
		T_p2 value=delta_mask.value[0];		//����
		T_p2 deno=delta_mask.denominator[0];	//��ĸ
		if(value==0)
			continue;

		T_p1* S = (T_p1*)im_in_ptr[d_row]+d_col*im_ch;
		if(value==1)
		{
			for(j=0;j<line_sz;j++)
				D[j]=S[j];
		}
		else
		{
			for(j=0;j<line_sz;j++)
				D[j]=value*S[j];
		}

		if(deno!=1)
		{
			for(j=0;j<line_sz;j++)
				D[j]=D[j]/deno;
		}

		for( s=1;s<d_m_size;s++)	//������
		{
			d_row=delta_mask.row[s];
			d_col=delta_mask.col[s];
			value=delta_mask.value[s];			//����
			deno=delta_mask.denominator[s];	//��ĸ
			if(value==0)
				continue;

			S = (T_p1*)im_in_ptr[d_row]+d_col*im_ch;

			if(value==1)	//����Ϊ1
			{
				if(deno==1)	//��ĸΪ1
				{
					for(j=0;j<line_sz;j++)
						D[j]+=S[j];
				}
				else		//��ĸ��Ϊ1
				{
					for(j=0;j<line_sz;j++)
						D[j]+=S[j]/deno;
				}
			}
			else
			{
				if(deno==1)
				{
					for(j=0;j<line_sz;j++)
						D[j]+=value*S[j];
				}
				else
				{
					for(j=0;j<line_sz;j++)
						D[j]+=value*S[j]/deno;
				}
			}
		}
	}

	free(im_in_ptr);
	return true;
}

/*	2.3 �����˲�	
	���룺
	cv::Mat& image_in			����ͼ��(��ͨ�����ͨ������������)
	cv::Vec2i mask_size			ģ��ߴ�
	T_p2 mask_const				������
	�����
	cv::Mat& image_out			���ͼ��*/
template<typename T_p1,typename T_p2,typename T_p3> bool constMaskImageFiltering(cv::Mat& image_in,cv::Vec2i mask_size,T_p3 mask_const,cv::Mat& image_out)
{
	//1.��ȡ����ͼ���������
	int im_in_rows=image_in.rows,
		im_in_cols=image_in.cols;					//����
	int im_in_type=image_in.depth();				//����ͼ����������
	int im_ch=image_in.channels();					//ͼ��ͨ����
	int im_out_rows=im_in_rows-mask_size[0]+1,
		im_out_cols=im_in_cols-mask_size[1]+1;		//���ͼ������

	//2.�������ͼ��
	//T_p2 type_test;
	int im_out_type=CV_32F;
	if(typeid(T_p2)==typeid(uchar))
		im_out_type=CV_8U;
	else if(typeid(T_p2)==typeid(ushort))
		im_out_type=CV_16U;
	else if(typeid(T_p2)==typeid(short))
		im_out_type=CV_16S;
	else if(typeid(T_p2)==typeid(int))
		im_out_type=CV_32S;
	else if(typeid(T_p2)==typeid(float))
		im_out_type=CV_32F;

	image_out.create(im_out_rows,im_out_cols,CV_MAKETYPE(im_out_type,im_ch));

	//3.�����˲�
	cv::Mat middle_mat(im_in_rows,im_out_cols,image_out.type());

	for(int i=0;i<im_in_rows;i++)
	{
		rowConstFilter<T_p1,T_p2>(image_in.ptr<uchar>(i),middle_mat.ptr<uchar>(i),mask_size[1],im_out_cols,im_ch);
	}

	//4.�����˲�
	const uchar** im_m_ptr=(const uchar**) malloc(im_in_rows*sizeof(uchar*));;
	
	for(int i=0;i<im_in_rows;i++)
	{
		im_m_ptr[i]=middle_mat.ptr<uchar>(i);
	}

	colConstFilter<T_p2,T_p2>(im_m_ptr,image_out.data,mask_size[0],im_out_rows,im_out_cols,im_ch);
	
	//5.���Գ���
	T_p2 mask_const_2=mask_const;
	if(mask_const_2!=1)
	{
		if(mask_const_2==-1)
			image_out=-image_out;
		else
			image_out=image_out*mask_const_2;
	}


	free(im_m_ptr);

	return true;
}
//�з������˲�
template<typename T_p1,typename T_p2> void rowConstFilter(const uchar* src, uchar* dst,int xk_size,int width,int ch)
{
	const T_p1* S ;
	T_p2* D = (T_p2*)dst;				//�������ָ��
	int i , k;
	int width_ch=width*ch;
	int xk_w=xk_size*ch;	//ģ�帲�ǵĶ�ͨ���п�

	if(ch==3)	//���ô������Ĳ�������
	{
		S = (const T_p1*)src;		//��������ָ��
		T_p2 s0 = S[0], s1 = S[1], s2 =S[2];
		for( k = 1; k <xk_size; k++ ,S+=3)
		{
			s0 +=S[0]; s1 += S[1];s2 +=S[2]; 
		}
		D[0] = s0; D[1] = s1;D[2] = s2;

		S = (const T_p1*)src;		//��������ָ��

		D+=3;
		for( i=0 ; i < width_ch - 3; i += 3,S+=3,D+=3)
        {
			s0+=S[xk_w]-S[0];s1+=S[xk_w+1]-S[1];s2+=S[xk_w+2]-S[2];
            D[0] = s0; D[1] = s1;D[2] = s2;
        }
	}
	else
	{
		S = (const T_p1*)src;
		for(k=0;k<ch;k++,D++,S++)	//��ͨ������
		{
			T_p2 s0=S[0];
			for(i=ch;i<xk_w;i+=ch)
			{
				s0+=S[i];
			}
			D[0]=s0;
			for(i=0;i<width_ch-ch;i+=ch)
			{
				s0+=S[xk_w+i]-S[i];
				D[i+ch]=s0;
			}
		}
	}
}
//�����˲�
template<typename T_p1,typename T_p2> void colConstFilter(const uchar** src, uchar* dst,int yk_size, int nrows, int width,int ch)
{
	int i = 0,j, k;
	int width_ch=width*ch;

	T_p2* sums=(T_p2*) malloc((width_ch)*sizeof(T_p2));//�м����

	//�м������ֵ
	T_p2* D = (T_p2*)dst;				//�������ָ��
	const T_p1* S = (const T_p1*)src[0];	//��������ָ��
	for( j=0;j<width_ch;j++)
		sums[j]=S[j];

	for(k=1;k<yk_size-1;k++)
	{
		S = (const T_p1*)src[k];	//��������ָ��
		if(ch==3)	//���ô������Ĳ�������
		{
			for( j=0;j<=width_ch-3;j+=3,S+=3)
			{
				sums[j]+=S[0]; sums[j+1]+= S[1]; sums[j+2]+=S[2];
			}
		}
		else
		{
			for( j=0;j<=width_ch-3;j+=3,S+=3)
			{
				sums[j]+=S[0];
			}
		}
	}

	for( i=0;i<nrows;i++,src++)
	{
		D = (T_p2*)dst+i*width_ch;				//�������ָ��

		if(ch==3)	//���ô������Ĳ�������
		{ 
			const T_p1* S_d = (const T_p1*)src[0];
			const T_p1* S_u = (const T_p1*)src[yk_size-1];
			T_p2 v_0,v_1,v_2;

			for(j=0;j<=width_ch-3;j+=3,S_d+=3,S_u+=3,D+=3)
			{
				v_0=sums[j]+S_u[0];v_1=sums[j+1]+S_u[1];v_2=sums[j+2]+S_u[2];
				D[0]=v_0; D[1]=v_1; D[2]=v_2;
				sums[j]=v_0-S_d[0];sums[j+1]=v_1-S_d[1];sums[j+2]=v_2-S_d[2];
			}
		}
		else
		{
			const T_p1* S_d = (const T_p1*)src[0];
			const T_p1* S_u = (const T_p1*)src[yk_size-1];
			T_p2 v_0;

			for(j=0;j<width_ch;j++)
			{
				v_0=sums[j]+S_u[j];
				D[j]=v_0;
				sums[j]=v_0-S_d[j];
			}
		}
	}
	free(sums);
}

/*	2.4 x,y����ģ���˲�	
	���룺
	cv::Mat& image_in			����ͼ��(��ͨ�����ͨ������������)
	Vector_Mask& x_filter		xģ������
	Vector_Mask& y_filter		yģ������
	Int_Mask_Divide& int_mask	����ģ��
	const string data_type		���ͼ���������ͣ�uchar char ushort short int float,Ĭ����image_in������ͬ��
	�����
	cv::Mat& image_out			���ͼ��*/
bool vectorMaskImageFiltering(cv::Mat& image_in,Vector_Mask& x_filter,Vector_Mask& y_filter,Int_Mask_Divide& int_mask,const string data_type,cv::Mat& image_out)
{
	//float t_us=(float)clock();//test

	//1.�ж�����ͼ������,׼����ز���
	int im_in_depth=image_in.depth();
	bool flag=false;

	cv::Vec2i mask_size=cv::Vec2i(y_filter.size,x_filter.size);	//�˲�ģ��ߴ�


	//2.�����������ͼ�����͵���ģ�庯��
	if(data_type=="uchar"||data_type=="char"||data_type=="ushort"||data_type=="short"||data_type=="int"||im_in_depth<CV_32S&&data_type=="")	//�����������
	{
		//float t_start=clock();//test

		//�����������͹�һ��
		cv::Mat image_in_co;
		if(im_in_depth==CV_32S)
			image_in_co=image_in;
		else
			image_in.convertTo(image_in_co,CV_32S);

		//׼������
		vector<int> x_filter_common=int_mask.vector_x.basic_vector,
			x_filter_symmetry=int_mask.vector_x.symmetry_vector,
			y_filter_common=int_mask.vector_y.basic_vector,
			y_filter_symmetry=int_mask.vector_y.symmetry_vector;

		int delta_x_size=int_mask.vector_x.delta_mask.size(),
			delta_y_size=int_mask.vector_y.delta_mask.size();
		DividePointL<int> div_px,div_py;

		div_px.id.resize(delta_x_size);
		div_px.value.resize(delta_x_size);
		for(int i=0;i<delta_x_size;i++)
		{
			div_px.id[i]=int_mask.vector_x.delta_mask[i][0];
			div_px.value[i]=int_mask.vector_x.delta_mask[i][1];
		}
		div_py.id.resize(delta_y_size);
		div_py.value.resize(delta_y_size);
		for(int i=0;i<delta_y_size;i++)
		{
			div_py.id[i]=int_mask.vector_y.delta_mask[i][0];
			div_py.value[i]=int_mask.vector_y.delta_mask[i][1];
		}

		int x_const=int_mask.vector_x.filter_constant,
			x_lnr=int_mask.vector_x.filter_linear,
			x_p2=int_mask.vector_x.filter_p2,
			x_deno=int_mask.vector_x.denominator,
			xf_s=int_mask.vector_x.state;

		int y_const=int_mask.vector_y.filter_constant,
			y_lnr=int_mask.vector_y.filter_linear,
			y_deno=int_mask.vector_y.denominator,
			y_p2=int_mask.vector_y.filter_p2,
			yf_s=int_mask.vector_y.state;

		image_out.create(image_in.rows-mask_size[0]+1,image_in.cols-mask_size[1]+1,CV_MAKETYPE(CV_32S,image_in.channels()));

		//float t_used = (clock()-t_start)/1000.0;//test
		//cout<<"vectorMaskImageFiltering:����׼����ʱ"<<t_used<<"s"<<endl;//test

		//����ģ�庯������
		flag=vectorMaskImageFilter<int,int>(image_in_co,image_out,mask_size,
							x_filter_common,x_filter_symmetry,
							div_px, x_const, x_lnr, x_p2, x_deno ,int_mask.vector_x.state,
							y_filter_common,y_filter_symmetry,
							div_py, y_const, y_lnr, y_p2, y_deno,int_mask.vector_y.state);

		if(flag==false)
		{
			cout<<"error: in vectorMaskImageFiltering->"<<endl;
			return false;
		}
	}
	else if(data_type=="float"||im_in_depth==CV_32F&&data_type=="")	//�������������
	{
		//float t_start=(float)clock();//test
		//�����������͹�һ��
		cv::Mat image_in_co;
		if(im_in_depth==CV_32F)
			image_in_co=image_in;
		else
			image_in.convertTo(image_in_co,CV_32F);

		//׼������
		vector<float> x_filter_common=x_filter.basic_vector,
			x_filter_symmetry=x_filter.symmetry_vector,
			y_filter_common=y_filter.basic_vector,
			y_filter_symmetry=y_filter.symmetry_vector;

		int delta_x_size=x_filter.delta_mask.size(),
			delta_y_size=y_filter.delta_mask.size();
		DividePointL<float> div_px,div_py;

		div_px.id.resize(delta_x_size);
		div_px.value.resize(delta_x_size);
		for(int i=0;i<delta_x_size;i++)
		{
			div_px.id[i]=(int)x_filter.delta_mask[i][0];
			div_px.value[i]=(int)x_filter.delta_mask[i][1];
		}
		div_py.id.resize(delta_y_size);
		div_py.value.resize(delta_y_size);
		for(int i=0;i<delta_y_size;i++)
		{
			div_py.id[i]=(int)y_filter.delta_mask[i][0];
			div_py.value[i]=(int)y_filter.delta_mask[i][1];
		}

		float x_const=x_filter.filter_constant,
			x_lnr=x_filter.filter_linear,
			x_p2=x_filter.filter_p2,
			x_deno=1;
		int xf_s=x_filter.state;

		float y_const=y_filter.filter_constant,
			y_lnr=y_filter.filter_linear,
			y_p2=y_filter.filter_p2,
			y_deno=1;
		int yf_s=y_filter.state;

		image_out.create(image_in.rows-mask_size[0]+1,image_in.cols-mask_size[1]+1,CV_MAKETYPE(CV_32F,image_in.channels()));

		//float t_used = (clock()-t_start)/1000.0;//test
		//cout<<"vectorMaskImageFiltering:����׼����ʱ"<<t_used<<"s"<<endl;//test

		//����ģ�庯������
		flag=vectorMaskImageFilter<float,float>(image_in_co,image_out,mask_size,
							x_filter_common,x_filter_symmetry,
							div_px, x_const, x_lnr, x_p2, x_deno ,int_mask.vector_x.state,
							y_filter_common,y_filter_symmetry,
							div_py, y_const, y_lnr, y_p2, y_deno,int_mask.vector_y.state);

		if(flag==false)
		{
			cout<<"error: in vectorMaskImageFiltering->"<<endl;
			return false;
		}
	}
	else
	{
		cout<<"error: IN vectorMaskImageFiltering: ���ͼ������������Ч"<<endl;
		return false;
	}


	//3.���ͼ��ı�Ϊָ������
	if(data_type=="uchar")
	{
		image_out.convertTo(image_out,CV_8U);
	}
	else if(data_type=="char")
	{
		image_out.convertTo(image_out,CV_8U);
	}
	else if(data_type=="ushort")
	{
		image_out.convertTo(image_out,CV_16U);
	}
	else if(data_type=="short")
	{
		image_out.convertTo(image_out,CV_16S);
	}
	else if(data_type=="int")
	{
		image_out.convertTo(image_out,CV_32S);
	}
	else if(data_type=="float")
	{
		image_out.convertTo(image_out,CV_32F);
	}
	else if(data_type=="")	//������ͼ��������ͬ
	{
		image_out.convertTo(image_out,im_in_depth);
	}
	//t_us = (clock()-t_us)/1000.0;//test
	//cout<<"vectorMaskImageFiltering ��ʱ��"<<t_us<<"s"<<endl;//test

	return true;
}
//���зֽ��˲�ģ�庯��
template<typename T_p1,typename T_p2,typename T_p3>
bool vectorMaskImageFilter(cv::Mat& image_in,cv::Mat& image_out,cv::Vec2i mask_size,
							vector<T_p3>& x_filter_common,vector<T_p3>& x_filter_symmetry,
							DividePointL<T_p3>& div_px,
							T_p3 x_const,T_p3 x_lnr,T_p3 x_p2,T_p3 x_deno ,int xf_s,
							vector<T_p3>& y_filter_common,vector<T_p3>& y_filter_symmetry,
							DividePointL<T_p3>& div_py,
							T_p3 y_const,T_p3 y_lnr,T_p3 y_p2,T_p3 y_deno,int yf_s)
{
	//1.��ȡ����ͼ���������
	int im_in_rows=image_in.rows,
		im_in_cols=image_in.cols;					//����
	int im_in_type=image_in.depth();				//����ͼ����������
	int im_ch=image_in.channels();					//ͼ��ͨ����
	int im_out_rows=im_in_rows-mask_size[0]+1,
		im_out_cols=im_in_cols-mask_size[1]+1;		//���ͼ������

	//2.�������ͼ��
	//T_p2 type_test=T_p2(0);
	int im_out_type=CV_32F;
	if(typeid(T_p2)==typeid(uchar))
		im_out_type=CV_8U;
	else if(typeid(T_p2)==typeid(ushort))
		im_out_type=CV_16U;
	else if(typeid(T_p2)==typeid(short))
		im_out_type=CV_16S;
	else if(typeid(T_p2)==typeid(int))
		im_out_type=CV_32S;
	else if(typeid(T_p2)==typeid(float))
		im_out_type=CV_32F;

	if(image_out.rows!=im_out_rows||image_out.cols!=im_out_cols||image_out.depth()!=im_out_type)
		image_out.create(im_out_rows,im_out_cols,CV_MAKETYPE(im_out_type,im_ch));

	cv::Mat im_middle(im_in_rows,im_out_cols,CV_MAKETYPE(im_out_type,im_ch));
	//cv::Mat div_m_x;		//x�����������˲����
	//cv::Mat div_m_y;		//y�����������˲����

	//3.x�����˲�
	//float t_start=(float)clock();//test
	if(xf_s==0)			//��ͨ�˲�
	{
		cv::Mat x_filter(x_filter_common);
		for(int i=0;i<im_in_rows;i++)
			rowCommonFilter<T_p1,T_p2>(image_in.ptr<uchar>(i),im_middle.ptr<uchar>(i),x_filter.data,mask_size[1],im_out_cols,im_ch);
	}
	else if(xf_s==1)	//�Գ��˲�
	{
		cv::Mat x_filter(x_filter_symmetry);
		for(int i=0;i<im_in_rows;i++)
			rowSymmetryFilter<T_p1,T_p2>(image_in.ptr<uchar>(i),im_middle.ptr<uchar>(i),x_filter.data,mask_size[1],im_out_cols,im_ch);
	}
	else if(xf_s==2)	//�Ȳ��˲�
	{
		if(x_lnr==0)	//������Ϊ0�����ó����˲�
		{
			if(x_const==0)	//������ҲΪ0������0����
			{
				im_middle=0*im_middle;
			}
			else
			{
				for(int i=0;i<im_in_rows;i++)
					rowConstFilter<T_p1,T_p2>(image_in.ptr<uchar>(i),im_middle.ptr<uchar>(i),mask_size[1],im_out_cols,im_ch);

			}
		}
		else	//�����Ϊ0�����������˲�
		{
			for(int i=0;i<im_in_rows;i++)
				rowGradeFilter<T_p1,T_p2>(image_in.ptr<uchar>(i),im_middle.ptr<uchar>(i),(T_p2)x_const,(T_p2) x_lnr,mask_size[1],im_out_cols,im_ch);
		}
	}
	else if(xf_s==3)	//�������˲�
	{
		if(x_lnr==0&&x_p2==0)	//����������Ϊ0�����ó����˲�
		{
			if(x_const==0)	//������ҲΪ0������0����
			{
				im_middle=0*im_middle;
			}
			else
			{
				for(int i=0;i<im_in_rows;i++)
					rowConstFilter<T_p1,T_p2>(image_in.ptr<uchar>(i),im_middle.ptr<uchar>(i),mask_size[1],im_out_cols,im_ch);

			}
		}
		else	//�����Ϊ0�������������˲�
		{
			for(int i=0;i<im_in_rows;i++)
				rowParabolaFilter<T_p1,T_p2>(image_in.ptr<uchar>(i),im_middle.ptr<uchar>(i),(T_p2)x_const,(T_p2) x_lnr,(T_p2) x_p2,mask_size[1],im_out_cols,im_ch);
		}
	}
	//x�������˲�
	if(div_px.id.size()>0)
	{
		//div_m_x=cv::Mat::zeros(im_in_rows,im_out_cols,CV_MAKETYPE(im_out_type,im_ch));

		for(int i=0;i<im_in_rows;i++)
			rowDivideFilter<T_p1,T_p2>(image_in.ptr<uchar>(i),im_middle.ptr<uchar>(i),div_px,im_out_cols,im_ch);

		//x�����˲�����ϲ�
		//im_middle=im_middle+div_m_x;
	}
	//float t_used = ((float)clock()-t_start)/1000.0;//test
	//cout<<"vectorMaskImageFilter:x�����˲���ʱ"<<t_used<<"s"<<endl;//test
	//cv::imshow("im_middle",im_middle/255);cv::waitKey();//TEST

	bool x_demo_flag=false;
	if(x_deno!=1&&typeid(T_p2)!=typeid(float)&&log(x_deno*y_deno)/log(16)+3.75>sizeof(T_p2))	//��ֹ�������
	{
		im_middle=im_middle/(T_p2)x_deno;
		x_demo_flag=true;
	}


	//4.y�����˲�
	//t_start=(float)clock();//test

	const uchar** im_m_ptr=(const uchar**) malloc(im_in_rows*sizeof(uchar*));;
	for(int i=0;i<im_in_rows;i++)
		im_m_ptr[i]=im_middle.ptr<uchar>(i);

	if(yf_s==0)			//��ͨ�˲�
	{
		cv::Mat y_filter(y_filter_common);
		colCommonFilter<T_p2,T_p2>(im_m_ptr,image_out.data,y_filter.data,mask_size[0],im_out_rows,im_out_cols,im_ch);
	}
	else if(yf_s==1)	//�Գ��˲�
	{
		cv::Mat y_filter(y_filter_symmetry);
		colSymmetryFilter<T_p2,T_p2>(im_m_ptr,image_out.data,y_filter.data,mask_size[0],im_out_rows,im_out_cols,im_ch);
	}
	else if(yf_s==2)	//�Ȳ��˲�
	{
		if(y_lnr==0)	//������Ϊ0�����ó����˲�
		{
			if(y_const==0)	//������ҲΪ0������0����
			{
				image_out=0*image_out;
			}
			else
			{
				colConstFilter<T_p2,T_p2>(im_m_ptr,image_out.data,mask_size[0],im_out_rows,im_out_cols,im_ch);
			}
		}
		else	//�����Ϊ0�����������˲�
		{
			colGradeFilter<T_p2,T_p2>(im_m_ptr,image_out.data,(T_p2)y_const,(T_p2) y_lnr,mask_size[0],im_out_rows,im_out_cols,im_ch);
		}
	}
	else if(yf_s==3)	//�������˲�
	{
		if(y_lnr==0&&y_p2==0)	//����������Ϊ0�����ó����˲�
		{
			if(y_const==0)	//������ҲΪ0������0����
			{
				image_out=0*image_out;
			}
			else
			{
				colConstFilter<T_p2,T_p2>(im_m_ptr,image_out.data,mask_size[0],im_out_rows,im_out_cols,im_ch);
			}
		}
		else	//��Ϊ0�������������˲�
		{
			colParabolaFilter<T_p2,T_p2>(im_m_ptr,image_out.data,(T_p2)y_const,(T_p2) y_lnr,(T_p2) y_p2,mask_size[0],im_out_rows,im_out_cols,im_ch);
		}
	}
	//y�������˲�
	if(div_py.id.size()>0)
	{
		//div_m_y=cv::Mat::zeros(im_out_rows,im_out_cols,CV_MAKETYPE(im_out_type,im_ch));
		colDivideFilter<T_p2,T_p2>(im_m_ptr,image_out.data,div_py,im_out_rows,im_out_cols,im_ch);

		//y�����˲�����ϲ�
		//image_out=image_out+div_m_y;
	}
	//t_used = ((float)clock()-t_start)/1000.0;//test
	//cout<<"vectorMaskImageFilter:y�����˲���ʱ"<<t_used<<"s"<<endl;//test

	if(x_demo_flag)	//֮ǰ�Ѿ�����x
	{
		if(y_deno!=1)
			image_out=image_out/(T_p2)y_deno;
	}
	else
	{
		if(x_deno*y_deno!=1)
			image_out=image_out/((T_p2)(x_deno*y_deno));
	}

	free(im_m_ptr);
	return true;
}

/*	2.4.1 x,y��ͨ����ģ���˲�
	*/
/*	�����˲�
	���룺
	const uchar* src			����ͼ��һ��
	const uchar* xkernal		x����ģ��ָ��
	int xk_size					xģ�峤��
	int width					���ͼ����
	int ch						ͼ��ͨ����
	�����
	uchar* dst					���ͼ��һ��
*/
template<typename T_p1,typename T_p2> void rowCommonFilter(const uchar* src, uchar* dst,const uchar* xkernal,int xk_size,int width,int ch)
{
	const T_p1* S ;
	T_p2* D = (T_p2*)dst;				//�������ָ��
	T_p2* K_p = (T_p2*)xkernal;			//�˲���ָ��
	int i = 0, k;
	width*=ch;

	if(ch==3)	//���ô������Ĳ�������
	{
		for( ; i <= width - 3; i += 3 )
        {
			S = (const T_p1*)src+i;		//��������ָ��
			T_p2 f = K_p[0];
            T_p2 s0 = f*S[0], s1 = f*S[1], s2 =f*S[2];

            for( k = 1; k < xk_size; k++,S+=3)
            {
				f = K_p[k];
                s0 +=f*S[0]; s1 += f*S[1];s2 +=f*S[2]; 
            }

            D[i] = s0; D[i+1] = s1;D[i+2] = s2;
        }
	}
	else
	{
		for( ;i<width;i++)
		{
			S = (const T_p1*)src+i;
			T_p2 s0=S[0]*K_p[0];
			for(k=1;k<xk_size;k++,S += ch)
			{
				s0+=S[0]*K_p[k];
			}
			D[i]=s0;
		}
	}
}
/*	�����˲�
	���룺
	const uchar** src			����ͼ��Ķ�άָ��
	const uchar* ykernal		y����ģ��ָ��
	int yk_size					yģ�峤��
	int nrows					���ͼ������
	int width					���ͼ����
	int ch						ͼ��ͨ����
	�����
	uchar* dst					���ͼ��Ķ�άָ��*/
template<typename T_p1,typename T_p2> void colCommonFilter(const uchar** src, uchar* dst,const uchar* ykernal,int yk_size, int nrows, int width,int ch)
{
	T_p2* K_p = (T_p2*)ykernal;			//�˲���ָ��
	int i = 0,j, k;
	width*=ch;

	T_p2* sums=(T_p2*) malloc((width)*sizeof(T_p2));

	for( ;i<nrows;i++,src++)
	{
		T_p2* D = (T_p2*)dst+i*width;				//�������ָ��

		//�м������ֵ
		for( j=0;j<width;j++)
			sums[j]=0;

		if(ch==3)	//���ô������Ĳ�������
		{ 
			const T_p1* S = (const T_p1*)src[0];
			T_p2 kn_v=K_p[0];	//ÿ�е�ykernalֵ��ͬ
			for(j=0;j<=width-3;j+=3,S+=3)
			{
				sums[j]= kn_v*S[0]; sums[j+1] = kn_v*S[1]; sums[j+2] = kn_v*S[2];
			}

			for(k=1;k<yk_size;k++)
			{
				S = (const T_p1*)src[k];	//��������ָ��
				kn_v=K_p[k];	//ÿ�е�ykernalֵ��ͬ

				for( j=0;j<=width-3;j+=3,S+=3)
				{
					sums[j]+=kn_v*S[0]; sums[j+1]+= kn_v*S[1]; sums[j+2]+=kn_v*S[2];
				}
			}
			for( j=0;j<=width-3;j+=3)
			{
				D[j]=sums[j];D[j+1]=sums[j+1];D[j+2]=sums[j+2];
			}
		}
		else
		{
			const T_p1* S = (const T_p1*)src[0];
			T_p2 kn_v=K_p[0];	//ÿ�е�ykernalֵ��ͬ
			for( j=0;j<width;j++)
			{
				sums[j]= kn_v*S[j];
			}

			for(k=1;k<yk_size;k++)
			{
				S = (const T_p1*)src[k];	//��������ָ��
				T_p2 kn_v=K_p[k];	//ÿ�е�ykernalֵ��ͬ

				for( j=0;j<width;j++)
				{
					sums[j]+= kn_v*S[j];
				}
			}
			for( j=0;j<width;j++)
			{
				D[j]=sums[j];
			}
		}
	}
	free(sums);
}
/*	2.4.2 x,y�Գ�����ģ���˲�
	*/
/*	�����˲�
	���룺
	const uchar* src			����ͼ��һ��
	const uchar* xkernal		x����ģ��ָ��
	int xk_size					xģ�峤��
	int width					���ͼ����
	int ch						ͼ��ͨ����
	�����
	uchar* dst					���ͼ��һ��
*/
template<typename T_p1,typename T_p2> void rowSymmetryFilter(const uchar* src, uchar* dst,const uchar* xkernal,int xk_size,int width,int ch)
{
	const T_p1* S ;
	T_p2* D = (T_p2*)dst;				//�������ָ��
	T_p2* K_p = (T_p2*)xkernal;			//�˲���ָ��
	int x_cent=(xk_size-1)/2;			//�˲��������ĵ�����
	K_p+=x_cent;						//ģ��ָ��ָ����������

	bool xk_odd=true;					//ģ������Ԫ�ظ�����ż��

	//�ж��˲�ģ����ż��
	if(xk_size%x_cent==0)	//����Ϊ0��ż��
		xk_odd=false;

	int i = 0, k,j;
	width*=ch;

	//int test1,test=x_cent*3;//test
	if(xk_odd)//ģ�峤��Ϊ����
	{
		if(ch==3)	//���ô������Ĳ�������
		{
			S = (const T_p1*)src+x_cent*3;		//��������ָ��
			for( ; i <= width - 3; i += 3,S+=3)
			{
				T_p2& f = K_p[0];
				T_p2 s0 = f*S[0], s1 = f*S[1], s2 =f*S[2];

				//test1=test;
				for( k = 1,j=ch; k <=x_cent; k++ ,j+=ch)
				{
					T_p2& f1 = K_p[k];
					s0 +=f1*(S[j]+S[-j]); s1 += f1*(S[j+1]+S[-j+1]);s2 +=f1*(S[j+2]+S[-j+2]); 
					//test1+=ch;
				}

				D[i] = s0; D[i+1] = s1;D[i+2] = s2;
				//test+=3;//test
			}
		}
		else
		{
			S = (const T_p1*)src+x_cent*3;		//��������ָ��
			for( ;i<width;i++,S++)
			{
				T_p2 s0=S[0]*K_p[0];
				for(k=1,j=ch;k<=x_cent;k++,j+=ch)
				{
					s0+=K_p[k]*(S[j]+S[-j]);
				}
				D[i]=s0;
			}
		}
	}
	else	//ģ�峤��Ϊż��
	{
		int j2;
		K_p+=1;
		if(ch==3)	//���ô������Ĳ�������
		{
			S = (const T_p1*)src+x_cent*3;		//��������ָ��
			for( ; i <= width - 3; i += 3,S+=3)
			{
				T_p2 f = K_p[0];
				T_p2 s0 = f*(S[0]+S[3]), s1 = f*(S[1]+S[4]), s2 =f*(S[2]+S[5]);

				for( k = 1,j=-ch,j2=2*ch; k <=x_cent; k++ ,j-=ch,j2+=ch)
				{
					f = K_p[k];
					s0 +=f*(S[j]+S[-j2]); s1 += f*(S[j+1]+S[j2+1]);s2 +=f*(S[j+2]+S[j2+2]); 
				}

				D[i] = s0; D[i+1] = s1;D[i+2] = s2;
			}
		}
		else
		{
			S = (const T_p1*)src+x_cent*ch;		//��������ָ��
			for( ;i<width;i++,S++)
			{
				T_p2 s0=(S[0]+S[1])*K_p[0];
				for(k=1,j=-ch,j2=2*ch;k<=x_cent;k++,j-=ch,j2+=ch)
				{
					s0+=K_p[k]*(S[j]+S[j2]);
				}
				D[i]=s0;
			}
		}
	}
	
}
/*	�����˲�
	���룺
	const uchar** src			����ͼ��Ķ�άָ��
	const uchar* ykernal		y����ģ��ָ��
	int yk_size					yģ�峤��
	int nrows					���ͼ������
	int width					���ͼ����
	int ch						ͼ��ͨ����
	�����
	uchar* dst					���ͼ��Ķ�άָ��*/
template<typename T_p1,typename T_p2> void colSymmetryFilter(const uchar** src, uchar* dst,const uchar* ykernal,int yk_size, int nrows, int width,int ch)
{
	T_p2* K_p = (T_p2*)ykernal;			//�˲���ָ��
	int y_cent=(yk_size-1)/2;			//�˲��������ĵ�����
	K_p+=y_cent;						//ģ��ָ��ָ����������

	bool yk_odd=true;					//ģ������Ԫ�ظ�����ż��

	//�ж��˲�ģ����ż��
	if(yk_size%y_cent==0)	//����Ϊ0��ż��
		yk_odd=false;

	int i = 0,j, k;
	width*=ch;

	T_p2* sums=(T_p2*) malloc((width)*sizeof(T_p2));

	if(yk_odd)//ģ�峤��Ϊ����
	{
		const T_p1* S,*S1;
		for( ;i<nrows;i++,src++)
		{
			T_p2* D = (T_p2*)dst+i*width;				//�������ָ��

			//�м������ֵ
			for( j=0;j<width;j++)
				sums[j]=0;

			if(ch==3)	//���ô������Ĳ�������
			{ 
				S = (const T_p1*)src[y_cent];
				T_p2 kn_v=K_p[0];	//ÿ�е�ykernalֵ��ͬ
				for(j=0;j<=width-3;j+=3)
				{
					sums[j]= kn_v*S[0]; sums[j+1] = kn_v*S[1]; sums[j+2] = kn_v*S[2];
					S+=3;
				}

				for(k=1;k<=y_cent;k++)
				{
					S = (const T_p1*)src[y_cent-k];	//��������ָ��
					S1 = (const T_p1*)src[y_cent+k];	//��������ָ��
					kn_v=K_p[-k];	//ÿ�е�ykernalֵ��ͬ

					for( j=0;j<=width-3;j+=3)//,S+=3,S1+=3)
					{
						S+=3;S1+=3;
						sums[j]+=kn_v*(S[0]+S1[0]); sums[j+1]+= kn_v*(S[1]+S1[1]); sums[j+2]+=kn_v*(S[2]+S1[2]);
					}
				}
				for( j=0;j<=width-3;j+=3)
				{
					D[j]=sums[j];D[j+1]=sums[j+1];D[j+2]=sums[j+2];
				}

			}
			else
			{
				S = (const T_p1*)src[y_cent];
				T_p2 kn_v=K_p[0];	//ÿ�е�ykernalֵ��ͬ
				for( j=0;j<width;j++)
				{
					sums[j]= kn_v*S[j];
				}

				for(k=1;k<y_cent;k++)
				{
					S = (const T_p1*)src[y_cent-k];	//��������ָ��
					S1 = (const T_p1*)src[y_cent+k];	//��������ָ��
					T_p2 kn_v=K_p[-k];	//ÿ�е�ykernalֵ��ͬ

					for( j=0;j<width;j++)
					{
						sums[j]+= kn_v*(S[j]+S1[j]);
					}
				}
				for( j=0;j<width;j++)
				{
					D[j]=sums[j];
				}
			}
		}
	}
	else//ģ�峤��Ϊż��
	{
		const T_p1* S,*S1;
		for( ;i<nrows;i++,src++)
		{
			T_p2* D = (T_p2*)dst+i*width;				//�������ָ��

			//�м������ֵ
			for( j=0;j<width;j++)
				sums[j]=0;

			if(ch==3)	//���ô������Ĳ�������
			{ 
				S = (const T_p1*)src[y_cent];
				S1= (const T_p1*)src[y_cent+1];
				T_p2 kn_v=K_p[0];	//ÿ�е�ykernalֵ��ͬ
				for(j=0;j<=width-3;j+=3)
				{
					sums[j]= kn_v*(S[0]+S1[0]); sums[j+1] = kn_v*(S[1]+S1[1]); sums[j+2] = kn_v*(S[2]+S1[2]);
					S+=3;
					S1+=3;
				}

				for(k=1;k<=y_cent;k++)
				{
					S = (const T_p1*)src[y_cent-k];	//��������ָ��
					S1 = (const T_p1*)src[y_cent+1+k];	//��������ָ��
					kn_v=K_p[-k];	//ÿ�е�ykernalֵ��ͬ

					for( j=0;j<=width-3;j+=3)
					{
						sums[j]+=kn_v*(S[0]+S1[0]); sums[j+1]+= kn_v*(S[1]+S1[1]); sums[j+2]+=kn_v*(S[2]+S1[2]);
						S+=3;
						S1+=3;
					}
				}
				for( j=0;j<=width-3;j+=3)
				{
					D[j]=sums[j];D[j+1]=sums[j+1];D[j+2]=sums[j+2];
				}

			}
			else
			{
				S = (const T_p1*)src[y_cent];
				S1 = (const T_p1*)src[y_cent+1];
				T_p2 kn_v=K_p[0];	//ÿ�е�ykernalֵ��ͬ
				for( j=0;j<width;j++)
				{
					sums[j]= kn_v*S[j];
				}

				for(k=1;k<y_cent;k++)
				{
					S = (const T_p1*)src[y_cent-k];	//��������ָ��
					S1 = (const T_p1*)src[y_cent+1+k];	//��������ָ��
					T_p2 kn_v=K_p[-k];	//ÿ�е�ykernalֵ��ͬ

					for( j=0;j<width;j++)
					{
						sums[j]+= kn_v*(S[j]+S1[j]);
					}
				}
				for( j=0;j<width;j++)
				{
					D[j]=sums[j];
				}
			}
		}
	}
	
	free(sums);
}

/*	2.4.3 x,y�Ȳ������˲�
*/
/*	�����˲�
	���룺
	const uchar* src			����ͼ��һ��
	T_p2 k_const				�˲�������
	T_p2 k_lnr					�˲�������
	int xk_size					xģ�峤��
	int width					���ͼ����
	int ch						ͼ��ͨ����
	�����
	uchar* dst					���ͼ��һ��
*/
template<typename T_p1,typename T_p2> void rowGradeFilter(const uchar* src, uchar* dst,T_p2 k_const,T_p2 k_lnr,int xk_size,int width,int ch)
{
	const T_p1* S ;
	T_p2* D = (T_p2*)dst;				//�������ָ��
	int i = 0,j, k;
	width*=ch;

	if(ch==3)	//���ô������Ĳ�������
	{
		//�Ȳ������ʼ��
		S = (const T_p1*)src;		//��������ָ��
		T_p2 tal_0=S[0],omiga_0=0,tal_1=S[1],omiga_1=0,tal_2=S[2],omiga_2=0;	//������
		S+=3;
		for( k = 1; k < xk_size; k++ ,S+=3)
		{
			//S+=ch;
			tal_0+=S[0]; tal_1+=S[1]; tal_2+=S[2];
			omiga_0+=k*S[0];omiga_1+=k*S[1];omiga_2+=k*S[2];
		}
		D[0] = tal_0*k_const+omiga_0*k_lnr;
		D[1] = tal_1*k_const+omiga_1*k_lnr;
		D[2] = tal_2*k_const+omiga_2*k_lnr;

		//�Ȳ��������
		S = (const T_p1*)src;
		const T_p1* S1=(const T_p1*)src+xk_size*3;
		for(i=3 ; i <= width - 3; i += 3,S+=3,S1+=3)
        {
			tal_0+=S1[0]-S[0]; tal_1+=S1[1]-S[1]; tal_2+=S1[2]-S[2];
			omiga_0+=xk_size*S1[0]-tal_0;omiga_1+=xk_size*S1[1]-tal_1;omiga_2+=xk_size*S1[2]-tal_2;

			D[i] = tal_0*k_const+omiga_0*k_lnr;
			D[i+1] = tal_1*k_const+omiga_1*k_lnr;
			D[i+2] = tal_2*k_const+omiga_2*k_lnr;
		}
	}
	else
	{
		for(k=0;k<ch;k++)	//��ͨ������
		{
			//������ֵ
			S = (const T_p1*)src+k;
			T_p2 tal_0=S[0],omiga_0=0;

			for( i = 1,j=ch; i < xk_size; i++ ,j+=ch)
			{
				tal_0+=S[j];
				omiga_0+=k*S[j];
			}
			D[k] = tal_0*k_const+omiga_0*k_lnr;

			//��������
			S = (const T_p1*)src+k;
			const T_p1* S1=(const T_p1*)src+xk_size*3+k;
			for(i=ch ; i <= width - ch; i += ch,S+=ch,S1+=ch)
			{
				tal_0+=S1[0]-S[0];
				omiga_0+=xk_size*S1[0]-tal_0;

				D[k+i] = tal_0*k_const+omiga_0*k_lnr;
			}
		}
	}
}
/*	�����˲�
	���룺
	const uchar** src			����ͼ��Ķ�άָ��
	T_p2 k_const				�˲�������
	T_p2 k_lnr					�˲�������
	int yk_size					yģ�峤��
	int nrows					���ͼ������
	int width					���ͼ����
	int ch						ͼ��ͨ����
	�����
	uchar** dst					���ͼ��Ķ�άָ��*/
template<typename T_p1,typename T_p2> void colGradeFilter(const uchar** src, uchar* dst,T_p2 k_const,T_p2 k_lnr,int yk_size, int nrows, int width,int ch)
{
	int i ,j;
	width*=ch;

	//��������һ�����ʼ��
	T_p2* tal_m=(T_p2*) malloc((width)*sizeof(T_p2));
	T_p2* omiga_m=(T_p2*) malloc((width)*sizeof(T_p2));

	const T_p1* S = (const T_p1*)src[0];
	for(j=0;j<width;j++)
	{
		tal_m[j]=S[j];
		omiga_m[j]=0;
	}
	for(i=1;i<yk_size;i++)
	{
		if(ch==3)	//���ô������Ĳ�������
		{
			S = (const T_p1*)src[i];
			int j1=1,j2=2;
			for(j=0;j<=width-3;j+=3,j1+=3,j2+=3)
			{
				tal_m[j]+=S[j]; tal_m[j1]+=S[j1]; tal_m[j2]+=S[j2];
				omiga_m[j]+=i*S[j];omiga_m[j1]+=i*S[j1]; omiga_m[j2]+=i*S[j2];
			}
		}
		else
		{
			const T_p1* S = (const T_p1*)src[i];
			for(j=0;j<width;j++)
			{
				tal_m[j]+=S[j];
				omiga_m[j]+=i*S[j];
			}
		}
	}

	//��������
	for(i=1;i<nrows;i++,src++)
	{
		T_p2* D = (T_p2*)dst+i*width;				//�������ָ��

		if(ch==3)	//���ô������Ĳ�������
		{ 
			S = (const T_p1*)src[0];
			const T_p1* S1 = (const T_p1*)src[yk_size];

			//T_p2 tal_0,tal_1,tal_2,om_0,om_1,om_2;

			int j1=1,j2=2;
			for(j=0;j<=width-3;j+=3,j1+=3,j2+=3,S+=3,S1+=3)
			{
				tal_m[j]=tal_m[j]+S1[0]-S[0];  tal_m[j1]=tal_m[j1]+S1[1]-S[1];  tal_m[j2]=tal_m[j2]+S1[2]-S[2];
				omiga_m[j]=omiga_m[j]+yk_size*S1[0]-tal_m[j];
				omiga_m[j1]=omiga_m[j1]+yk_size*S1[1]-tal_m[j1];
				omiga_m[j2]=omiga_m[j2]+yk_size*S1[2]-tal_m[j2];

				D[j] = tal_m[j]*k_const+omiga_m[j]*k_lnr;
				D[j1] = tal_m[j1]*k_const+omiga_m[j1]*k_lnr;
				D[j2] = tal_m[j2]*k_const+omiga_m[j2]*k_lnr;

				//tal_m[j]=tal_0; tal_m[j1]=tal_1; tal_m[j2]=tal_2;
				//omiga_m[j]=om_0; omiga_m[j1]=om_1;omiga_m[j2]=om_2;
			}
		}
		else
		{
			const T_p1* S = (const T_p1*)src[0];
			const T_p1* S1 = (const T_p1*)src[yk_size];

			for(j=0;j<width;j++)
			{
				tal_m[j]+=S1[j]-S[j];
				omiga_m[j]+=yk_size*S1[j]-tal_m[j];

				D[j] = tal_m[j]*k_const+omiga_m[j]*k_lnr;
			}
		}
	}
	free(tal_m);
	free(omiga_m);
}

/*	2.4.4 x,y������ģ���˲�
*/
/*	�����˲�
	���룺
	const uchar* src			����ͼ��һ��
	T_p2 k_const				�˲�������
	T_p2 k_lnr					�˲�������
	T_p2 k_p2					�˲�2����
	int xk_size					xģ�峤��
	int width					���ͼ����
	int ch						ͼ��ͨ����
	�����
	uchar* dst					���ͼ��һ��
*/
template<typename T_p1,typename T_p2> void rowParabolaFilter(const uchar* src, uchar* dst,T_p2 k_const,T_p2 k_lnr,T_p2 k_p2,int xk_size,int width,int ch)
{
	const T_p1* S ;
	T_p2* D = (T_p2*)dst;				//�������ָ��
	int i = 0,j, k;
	width*=ch;

	if(ch==3)	//���ô������Ĳ�������
	{
		//�Ȳ������ʼ��
		S = (const T_p1*)src;		//��������ָ��
		T_p2 tal_0=S[0],omiga_0=0,ve_0=0,tal_1=S[1],omiga_1=0,ve_1=0,tal_2=S[2],omiga_2=0,ve_2=0;	//������
		S+=3;
		for( k = 1; k < xk_size; k++ ,S+=3)
		{
			//S+=ch;
			tal_0+=S[0]; tal_1+=S[1]; tal_2+=S[2];
			omiga_0+=k*S[0];omiga_1+=k*S[1];omiga_2+=k*S[2];
			ve_0+=k*k*S[0];ve_1+=k*k*S[1];ve_2+=k*k*S[2];
		}
		D[0] = tal_0*k_const+omiga_0*k_lnr+ve_0*k_p2;
		D[1] = tal_1*k_const+omiga_1*k_lnr+ve_1*k_p2;
		D[2] = tal_2*k_const+omiga_2*k_lnr+ve_2*k_p2;

		//�Ȳ��������
		T_p1 xk_size_sqr=xk_size*xk_size;
		S = (const T_p1*)src;
		const T_p1* S1=(const T_p1*)src+xk_size*3;
		for(i=3 ; i <= width - 3; i += 3,S+=3,S1+=3)
        {
			T_p1 k2k=
			tal_0+=S1[0]-S[0]; tal_1+=S1[1]-S[1]; tal_2+=S1[2]-S[2];
			omiga_0+=xk_size*S1[0]-tal_0;omiga_1+=xk_size*S1[1]-tal_1;omiga_2+=xk_size*S1[2]-tal_2;
			ve_0+=xk_size_sqr*S1[0]-tal_0-2*omiga_0;ve_1+=xk_size_sqr*S1[1]-tal_1-2*omiga_1;ve_2+=xk_size_sqr*S1[2]-tal_2-2*omiga_2;

			D[i] = tal_0*k_const+omiga_0*k_lnr+ve_0*k_p2;
			D[i+1] = tal_1*k_const+omiga_1*k_lnr+ve_1*k_p2;
			D[i+2] = tal_2*k_const+omiga_2*k_lnr+ve_2*k_p2;
		}
	}
	else
	{
		T_p1 xk_size_sqr=xk_size*xk_size;

		for(k=0;k<ch;k++)	//��ͨ������
		{
			//������ֵ
			S = (const T_p1*)src+k;
			T_p2 tal_0=S[0],omiga_0=0,ve_0=0;

			for( i = 1,j=ch; i < xk_size; i++ ,j+=ch)
			{
				tal_0+=S[j];
				omiga_0+=k*S[j];
				ve_0+=k*k*S[j];
			}
			D[k] = tal_0*k_const+omiga_0*k_lnr+ve_0*k_p2;

			//��������
			S = (const T_p1*)src+k;
			const T_p1* S1=(const T_p1*)src+xk_size*3+k;
			for(i=ch ; i <= width - ch; i += ch,S+=ch,S1+=ch)
			{
				tal_0+=S1[0]-S[0];
				omiga_0+=xk_size*S1[0]-tal_0;
				ve_0+=xk_size_sqr*S1[0]-tal_0-2*omiga_0;

				D[k+i] = tal_0*k_const+omiga_0*k_lnr+ve_0*k_p2;
			}
		}
	}
}
/*	�����˲�
	���룺
	const uchar** src			����ͼ��Ķ�άָ��
	T_p2 k_const				�˲�������
	T_p2 k_lnr					�˲�������
	T_p2 k_p2					�˲�2����
	int yk_size					yģ�峤��
	int nrows					���ͼ������
	int width					���ͼ����
	int ch						ͼ��ͨ����
	�����
	uchar** dst					���ͼ��Ķ�άָ��*/
template<typename T_p1,typename T_p2> void colParabolaFilter(const uchar** src, uchar* dst,T_p2 k_const,T_p2 k_lnr,T_p2 k_p2,int yk_size, int nrows, int width,int ch)
{
	int i ,j;
	width*=ch;
	T_p1 yk_size_sqr=yk_size*yk_size;

	//��������1,2�����ʼ��
	T_p2* tal_m=(T_p2*) malloc((width)*sizeof(T_p2));
	T_p2* omiga_m=(T_p2*) malloc((width)*sizeof(T_p2));
	T_p2* p2_m=(T_p2*) malloc((width)*sizeof(T_p2));

	const T_p1* S = (const T_p1*)src[0];
	for(j=0;j<width;j++)
	{
		tal_m[j]=S[j];
		omiga_m[j]=0;
		p2_m[j]=0;
	}
	for(i=1;i<yk_size;i++)
	{
		if(ch==3)	//���ô������Ĳ�������
		{
			S = (const T_p1*)src[i];
			int j1=1,j2=2;
			for(j=0;j<=width-3;j+=3,j1+=3,j2+=3)
			{
				tal_m[j]+=S[j]; tal_m[j1]+=S[j1]; tal_m[j2]+=S[j2];
				omiga_m[j]+=i*S[j];omiga_m[j1]+=i*S[j1]; omiga_m[j2]+=i*S[j2];
				p2_m[j]+=i*i*S[j];p2_m[j1]+=i*i*S[j1];p2_m[j2]+=i*i*S[j2];
			}
		}
		else
		{
			const T_p1* S = (const T_p1*)src[i];
			for(j=0;j<width;j++)
			{
				tal_m[j]+=S[j];
				omiga_m[j]+=i*S[j];
				p2_m[j]+=i*i*S[j];
			}
		}
	}

	//��������
	for(i=1;i<nrows;i++,src++)
	{
		T_p2* D = (T_p2*)dst+i*width;				//�������ָ��

		if(ch==3)	//���ô������Ĳ�������
		{ 
			S = (const T_p1*)src[0];
			const T_p1* S1 = (const T_p1*)src[yk_size];

			//T_p2 tal_0,tal_1,tal_2,om_0,om_1,om_2;

			int j1=1,j2=2;
			for(j=0;j<=width-3;j+=3,j1+=3,j2+=3,S+=3,S1+=3)
			{
				tal_m[j]=tal_m[j]+S1[0]-S[0];  tal_m[j1]=tal_m[j1]+S1[1]-S[1];  tal_m[j2]=tal_m[j2]+S1[2]-S[2];
				omiga_m[j]=omiga_m[j]+yk_size*S1[0]-tal_m[j];
				omiga_m[j1]=omiga_m[j1]+yk_size*S1[1]-tal_m[j1];
				omiga_m[j2]=omiga_m[j2]+yk_size*S1[2]-tal_m[j2];
				p2_m[j]=p2_m[j]+yk_size_sqr*S1[0]-tal_m[j]-2*omiga_m[j];
				p2_m[j1]=p2_m[j1]+yk_size_sqr*S1[1]-tal_m[j1]-2*omiga_m[j1];
				p2_m[j2]=p2_m[j2]+yk_size_sqr*S1[2]-tal_m[j2]-2*omiga_m[j2];

				D[j] = tal_m[j]*k_const+omiga_m[j]*k_lnr+p2_m[j]*k_p2;
				D[j1] = tal_m[j1]*k_const+omiga_m[j1]*k_lnr+p2_m[j1]*k_p2;
				D[j2] = tal_m[j2]*k_const+omiga_m[j2]*k_lnr+p2_m[j2]*k_p2;

				//tal_m[j]=tal_0; tal_m[j1]=tal_1; tal_m[j2]=tal_2;
				//omiga_m[j]=om_0; omiga_m[j1]=om_1;omiga_m[j2]=om_2;
			}
		}
		else
		{
			const T_p1* S = (const T_p1*)src[0];
			const T_p1* S1 = (const T_p1*)src[yk_size];

			for(j=0;j<width;j++)
			{
				tal_m[j]+=S1[j]-S[j];
				omiga_m[j]+=yk_size*S1[j]-tal_m[j];
				p2_m[j]=p2_m[j]+yk_size_sqr*S1[j]-tal_m[j]-2*omiga_m[j];

				D[j] = tal_m[j]*k_const+omiga_m[j]*k_lnr+p2_m[j]*k_p2;
			}
		}
	}
	free(tal_m);
	free(omiga_m);
}

/*	2.4.5 x,y�����������˲�
*/
/*	�����˲�
	���룺
	const uchar* src			����ͼ��һ��
	DividePointL<T_p2>& div_px	���˲���������
	int width					���ͼ����
	int ch						ͼ��ͨ����
	�����
	uchar* dst					���ͼ��һ��
*/
template<typename T_p1,typename T_p2,typename T_p3> void rowDivideFilter(const uchar* src, uchar* dst, DividePointL<T_p3>& div_px,int width,int ch)
{
	const T_p1* S ;
	T_p2* D = (T_p2*)dst;				//�������ָ��
	int i = 0,s=0;
	width*=ch;

	//��ʼ����ÿ���������˲�
	int div_p_n=div_px.id.size();	//���������
	for(;s<div_p_n;s++)
	{
		int div_id=div_px.id[s];	//������λ��
		const T_p2 div_v=div_px.value[s];	//����ֵ
		if(div_v==0)
			continue;

		S = (const T_p1*)src+div_id;		//��������ָ��
		for(; i <width ; i++,S++,D++)
		{
			D[0]+=S[0]*div_v;
		}
	}
}
/*	�����˲�
	���룺
	const uchar** src			����ͼ��Ķ�άָ��
	DividePointL<T_p2>& div_py	���˲���������
	int nrows					���ͼ������
	int width					���ͼ����
	int ch						ͼ��ͨ����
	�����
	uchar** dst					���ͼ��Ķ�άָ��*/
template<typename T_p1,typename T_p2,typename T_p3> void colDivideFilter(const uchar** src, uchar* dst, DividePointL<T_p3>& div_py, int nrows, int width,int ch)
{
	int i ,j,s=0;
	width*=ch;

	//��ʼ����ÿ���������˲�
	int div_p_n=div_py.id.size();	//���������
	for(;s<div_p_n;s++)
	{
		int div_id=div_py.id[s];	//������λ��
		T_p2 div_v=div_py.value[s];	//����ֵ
		if(div_v==0)
			continue;

		for(i=1;i<nrows;i++)
		{
			const T_p1* S = (const T_p1*)src[div_id];	//��������ָ��
			T_p2* D = (T_p2*)dst+i*width;				//�������ָ��

			for(j=0; j <width ; j++,S++,D++)
			{
				D[0]+=S[0]*div_v;
			}
		}
	}
}


//�����˲����Գ���
bool fastImageFilterTest()
{
	//1.����ͼ��
	cv::Mat image_in;
	if(_waccess(L"test images/Lena.jpg",0)==0)
		image_in=cv::imread("test images/Lena.jpg");
	else if(_waccess(L"../test images/Lena.jpg",0)==0)
		image_in=cv::imread("../test images/Lena.jpg");
	else
		image_in=cv::imread("../../test images/Lena.jpg");

	cv::imshow("ԭͼ",image_in);
	cv::waitKey(400);

	/*double t_start0=(double)cv::getTickCount();
	image_in.convertTo(image_in,CV_32F);
	double t_used0 = ((double)cv::getTickCount()-t_start0)/cv::getTickFrequency();
	cout<<"ͼ��ת����ʱ"<<t_used0<<"s"<<endl;*/

	//2.����ģ��
	cv::Mat filter_mask=0*cv::Mat::ones(51,51,CV_32FC1)/4000;
	float sum_kernel;
	sum_kernel=0;
	for(int i=0;i<51;i++)
	{
		for(int j=0;j<51;j++)
		{
			filter_mask.at<float>(i,j)=0.1*(i-25)*(j-25);
			sum_kernel+=filter_mask.at<float>(i,j);
		}
	}
	filter_mask=filter_mask/sum_kernel;
	//filter_mask.at<float>(5,5)=-0.1;

	//3.���������
	float gamma=0.01;				//ģ��ֽ�����-�źű���ֵ
	bool filters_result_expand=true;//�˲�������Ե
	float standard_error;			//ģ��ֽ����
	cv::Mat image_out;				//���ͼ��
	bool result;
	bool flag_out;

	double t_start,t_used;

	cv::Vec3i rows_cols_channel;
	cv::Mat image_in_2;image_in.convertTo(image_in_2,CV_32F);

	cv::Vec2i mask_size;mask_size[0]=51;mask_size[1]=51;
	cv::Mat image_in_ex;
	imageExtension(image_in_2,mask_size, image_in_ex);

	//�Ȳ�ģ��
	FastImageMaskFilter filter_1;
	filter_1.inputMastFilter(filter_mask,0.1);

	t_start=(double)cv::getTickCount();
	filter_1.runFastImageMaskFilter(image_in_2,image_out);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();
	cout<<"float���ݣ��Ȳ�ģ��:"<<t_used<<"s"<<endl;

	cv::imshow("float���ݣ��Ȳ�ģ��",image_out/255);
	cv::waitKey(400);


	//cv::imshow("imageExtension",image_in_ex/255);cv::waitKey(10000);

	/*Vector_Mask filter_mask_x,filter_mask_y;
	vector<Point_Value> xy_delta_mask;
	filterMaskAnalysis(filter_mask, gamma,filter_mask_x,filter_mask_y,xy_delta_mask,standard_error,flag_out);
	cv::Mat middle_mat;
	xGradeImageFiltering(image_in,filter_mask_x,middle_mat);
	yGradeImageFiltering(middle_mat,filter_mask_y,image_out);

	cv::imshow("xy basic",image_out/255);cv::waitKey(10000);*/

	/*//�����˲�����
	float* image_in_p;//=imageTypeChange<float>(image_in_2,rows_cols_channel);
	float m_const=1.0/2500;

	t_start=(double)cv::getTickCount();
	constMaskImageFiltering<float,float>(image_in_ex,cv::Vec2i(51,51),m_const,image_out);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();

	cout<<"float���ݣ�����ģ��:"<<t_used<<"s"<<endl;

	cv::imshow("float���ݣ�����ģ��",image_out/255);cv::waitKey(100);*/

	/*//(1)int���ݣ���ģ��ֽ�
	t_start=(double)cv::getTickCount();
	//result=imageFiltering(image_in,filter_mask,gamma,0,0,filters_result_expand,standard_error,image_out);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();
	cout<<"��������ʱ->int���ݣ���ģ��ֽ�:"<<t_used<<"s"<<endl;

	//cv::imshow("��������ԣ�int���ݣ���ģ��ֽ�",image_out);

	//(2)int���ݣ�xyģ��ֽ�
	t_start=(double)cv::getTickCount();
	result=imageFiltering(image_in,filter_mask,gamma,0,1,filters_result_expand,standard_error,image_out);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();
	cout<<"��������ʱ->int���ݣ�xyģ��ֽ�:"<<t_used<<"s"<<endl;

	cv::imshow("��������ԣ�int���ݣ�xyģ��ֽ�",image_out);cv::waitKey(100);

	//(3)int���ݣ�xy,constģ��ֽ�
	t_start=(double)cv::getTickCount();
	result=imageFiltering(image_in,filter_mask,gamma,0,2,filters_result_expand,standard_error,image_out);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();
	cout<<"��������ʱ->int���ݣ�xy,constģ��ֽ�:"<<t_used<<"s"<<endl;

	cv::imshow("��������ԣ�int���ݣ�xy,constģ��ֽ�",image_out);

	while(true)
	{
		if(cv::waitKey(1000)>0)
			break;
	}

	//(4)float���ݣ���ģ��ֽ�
	t_start=(double)cv::getTickCount();
	//result=imageFiltering(image_in,filter_mask,gamma,1,0,filters_result_expand,standard_error,image_out);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();
	cout<<"��������ʱ->float���ݣ���ģ��ֽ�:"<<t_used<<"s"<<endl;

	//cv::imshow("��������ԣ�float���ݣ���ģ��ֽ�",image_out/255);
	//cv::waitKey(1000);

	//(5)float���ݣ�xyģ��ֽ�
	t_start=(double)cv::getTickCount();
	result=imageFiltering(image_in,filter_mask,gamma,1,1,filters_result_expand,standard_error,image_out);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();
	cout<<"��������ʱ->float���ݣ�xyģ��ֽ�:"<<t_used<<"s"<<endl;

	cv::imshow("��������ԣ�float���ݣ�xyģ��ֽ�",image_out/255);
	cv::waitKey(100);

	//(6)float���ݣ�xy,constģ��ֽ�
	t_start=(double)cv::getTickCount();
	result=imageFiltering(image_in,filter_mask,gamma,1,2,filters_result_expand,standard_error,image_out);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();
	cout<<"��������ʱ->float���ݣ�xy,constģ��ֽ�:"<<t_used<<"s"<<endl;

	cv::imshow("��������ԣ�float���ݣ�xy,constģ��ֽ�",image_out/255);
	cv::waitKey(100);*/

	/*while(true)
	{
		if(cv::waitKey(1000)>0)
			break;
	}*/

	//4.��opencv�ԱȲ���
	filter_mask=cv::Mat::zeros(51,51,CV_32FC1);
	sum_kernel=0;
	for(int i=0;i<51;i++)
	{
		for(int j=0;j<51;j++)
		{
			filter_mask.at<float>(i,j)=exp(-float((i-25)*(i-25)+(j-25)*(j-25))/512);
			sum_kernel+=filter_mask.at<float>(i,j);
		}
	}
	/*filter_mask=cv::Mat::zeros(1,51,CV_32FC1);
	sum_kernel=0;
	for(int i=0;i<1;i++)
	{
		for(int j=0;j<51;j++)
		{
			filter_mask.at<float>(i,j)=exp(-((i-0)*(i-0)+(j-25)*(j-25))/512.0);
			sum_kernel+=filter_mask.at<float>(i,j);
		}
	}*/
	filter_mask=filter_mask/sum_kernel;

	//float���ݣ�xyģ��ֽ�
	filter_1.inputMastFilter(filter_mask,0.1);
	t_start=(double)cv::getTickCount();
	filter_1.runFastImageMaskFilter(image_in_2,image_out);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();

	cout<<"��������ʱ->float���ݣ�xyģ��ֽ�:"<<t_used<<"s"<<endl;

	cv::imshow("��������ԣ���˹�ˣ�int���ݣ�xyģ��ֽ�",image_out/255);
	cv::waitKey(400);

	//float���ݣ�opencv
	cv::Mat image_out_g;
	t_start=(double)cv::getTickCount();
	cv::GaussianBlur(image_in_2,image_out_g,cv::Size(51,51),16,16);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();

	cout<<"��˹�ˣ�opencv:"<<t_used<<"s"<<endl;

	cv::imshow("��������ԣ���˹�ˣ�opencv",image_out_g/255);
	cv::waitKey(400);

	cv::Mat image_error=abs(image_out_g-image_out);
	cv::imshow("image_error",image_error/255);
	cv::waitKey(400);

	while(true)
	{
		if(cv::waitKey(1000)>0)
			break;
	}

	/*FastImageMaskFilter image_filter;
	image_filter.inputMastFilter(filter_mask,0.02);
	image_filter.runFastImageMaskFilter(image_in,image_out,1,true);*/

	//cv::imshow("FastImageMaskFilter���ԣ���˹�ˣ�float����",image_out/255);
	//cv::waitKey();

	return true;
}