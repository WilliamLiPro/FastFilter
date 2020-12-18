/***************************************************************
	>���ƣ�ͼ��Ļ�������
	>���ߣ���ά��
	>��ϵ��ʽ��williamli_pro@163.com
	>ʵ�ֶ�ͼ��Ŀ�����������
	>����Ҫ�㣺
	>1.uchar�Ͷ�ͨ��ͼ��Ŀ�����������
	>2.һ���ͨ��ͼ��Ŀ�����������

****************************************************************/

#include "imageIntegration.h"


/*	uchar(char)�Ͷ�ͨ��ͼ��Ŀ�����������
	���룺
	cv::Mat& image_in		����ͼ��uchar(char)�ͣ�
	int data_type			�������ͣ�0 uchar,1 char��
	�����
	cv::Mat& image_out		���ͼ��uchar�ͣ�
	����������falseʱ���������*/
bool fastUcharImageIntegration(cv::Mat& image_in,int data_type,cv::Mat& image_out)
{
	//1.�������ͼ��ߴ硢���͵Ȳ���
	int image_rows=image_in.rows;
	int image_cols=image_in.cols;

	int image_channels=image_in.channels();	//ͼ��ͨ����
	int image_data_type=image_in.depth();	//��ȡ����ͼ�����������

	if(data_type==0)	//uchar
	{
		if(image_data_type!=CV_8U)
		{
			cout<<"error: -> in fastUcharImageIntegration :����ͼ���������Ͳ���ȷ��Ӧ��Ϊuchar������"<<endl;
			return false;
		}
	}
	else
	{
		if(image_data_type!=CV_8S)
		{
			cout<<"error: -> in fastUcharImageIntegration :����ͼ���������Ͳ���ȷ��Ӧ��Ϊchar������"<<endl;
			return false;
		}
	}

	//2.�����������ͼ��
	image_out=cv::Mat::zeros(image_rows,image_cols,CV_MAKETYPE(CV_32S,image_channels));

	uchar* image_in_ptr_1;
	char* image_in_ptr_2;
	int* image_out_ptr_1;
	int* image_out_ptr_2;

	for(int i=0;i<image_rows;i++)
	{
		if(image_data_type==0)
			image_in_ptr_1=image_in.ptr<uchar>(i);
		else
			image_in_ptr_2=image_in.ptr<char>(i);

		image_out_ptr_1=image_out.ptr<int>(i);

		if(i>0)
		{
			image_out_ptr_2=image_out.ptr<int>(i-1);
		}

		if(image_data_type==0)	//uchar
		{
			for(int j=0;j<image_cols;j++)
			{
				for(int k=0;k<image_channels;k++)
				{
					image_out_ptr_1[j*image_channels+k]=image_in_ptr_1[j*image_channels+k];

					if(i>0)
					{
						image_out_ptr_1[j*image_channels+k]+=image_out_ptr_2[j*image_channels+k];
					}
					if(j>0)
					{
						image_out_ptr_1[j*image_channels+k]+=image_out_ptr_1[(j-1)*image_channels+k];
					}

					if(i>0&&j>0)
					{
						image_out_ptr_1[j*image_channels+k]-=image_out_ptr_2[(j-1)*image_channels+k];
					}
				}
			}
		}
		else
		{
			for(int j=0;j<image_cols;j++)
			{
				for(int k=0;k<image_channels;k++)
				{
					image_out_ptr_1[j*image_channels+k]=image_in_ptr_2[j*image_channels+k];

					if(i>0)
					{
						image_out_ptr_1[j*image_channels+k]+=image_out_ptr_2[j*image_channels+k];
					}
					if(j>0)
					{
						image_out_ptr_1[j*image_channels+k]+=image_out_ptr_1[(j-1)*image_channels+k];
					}

					if(i>0&&j>0)
					{
						image_out_ptr_1[j*image_channels+k]-=image_out_ptr_2[(j-1)*image_channels+k];
					}
				}
			}
		}
	}

	return true;
}

/*	һ���ͨ��ͼ��Ŀ�����������	
	���룺
	cv::Mat& image_in		����ͼ���������ͣ�
	�����
	cv::Mat& image_out		���ͼ��float�ͣ�
	����������falseʱ���������*/
bool fastNormalImageIntegration(cv::Mat& image_in,cv::Mat& image_out)
{
	//1.�������ͼ��ߴ硢���͵Ȳ���
	int image_rows=image_in.rows;
	int image_cols=image_in.cols;

	int image_channels=image_in.channels();	//ͼ��ͨ����
	int image_data_type=image_in.depth();	//��ȡ����ͼ�����������

	if(image_data_type!=CV_32F)
	{
		image_in.convertTo(image_in,CV_32F);
		return false;
	}

	//2.�����������ͼ��
	image_out=cv::Mat::zeros(image_rows,image_cols,image_in.type());

	float* image_in_ptr;
	float* image_out_ptr_1;
	float* image_out_ptr_2;

	for(int i=0;i<image_rows;i++)
	{
		image_in_ptr=image_in.ptr<float>(i);
		image_out_ptr_1=image_out.ptr<float>(i);

		if(i>0)
		{
			image_out_ptr_2=image_out.ptr<float>(i-1);
		}

		for(int j=0;j<image_cols;j++)
		{
			for(int k=0;k<image_channels;k++)
			{
				image_out_ptr_1[j*image_channels+k]=image_in_ptr[j*image_channels+k];

				if(i>0)
				{
					image_out_ptr_1[j*image_channels+k]+=image_out_ptr_2[j*image_channels+k];
				}
				if(j>0)
				{
					image_out_ptr_1[j*image_channels+k]+=image_out_ptr_1[(j-1)*image_channels+k];
				}

				if(i>0&&j>0)
				{
					image_out_ptr_1[j*image_channels+k]-=image_out_ptr_2[(j-1)*image_channels+k];
				}
			}
		}
	}
}

/*	ͼ�����������Ժ���	*/
bool imageIntegrationTest()
{
	cout<<endl<<"ͼ������������:"<<endl;

	cv::Mat image_in,image_out;
	image_in=cv::imread("����ͼƬ/Lena.jpg");
	cv::imshow("original image",image_in);

	double t_start=(double)cv::getTickCount();
	bool result=fastUcharImageIntegration(image_in,0,image_out);
	double t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();;
	cout<<"uchar������ʱ:"<<t_used<<"s"<<endl;
	
	cv::imshow("uchar",image_out/256);

	t_start=(double)cv::getTickCount();
	result=fastNormalImageIntegration(image_in,image_out);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();;
	cout<<"һ�������ʱ:"<<t_used<<"s"<<endl;
	
	cv::imshow("normal",image_out/256);

	cv::waitKey(100000);

	return true;
}
