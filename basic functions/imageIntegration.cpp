/***************************************************************
	>名称：图像的积分运算
	>作者：李维鹏
	>联系方式：williamli_pro@163.com
	>实现对图像的快速完整积分
	>技术要点：
	>1.uchar型多通道图像的快速完整积分
	>2.一般多通道图像的快速完整积分

****************************************************************/

#include "imageIntegration.h"


/*	uchar(char)型多通道图像的快速完整积分
	输入：
	cv::Mat& image_in		输入图像（uchar(char)型）
	int data_type			数据类型（0 uchar,1 char）
	输出：
	cv::Mat& image_out		输出图像（uchar型）
	当函数返回false时，程序出错*/
bool fastUcharImageIntegration(cv::Mat& image_in,int data_type,cv::Mat& image_out)
{
	//1.检测输入图像尺寸、类型等参数
	int image_rows=image_in.rows;
	int image_cols=image_in.cols;

	int image_channels=image_in.channels();	//图像通道数
	int image_data_type=image_in.depth();	//获取输入图像的数据类型

	if(data_type==0)	//uchar
	{
		if(image_data_type!=CV_8U)
		{
			cout<<"error: -> in fastUcharImageIntegration :输入图像数据类型不正确，应该为uchar型数据"<<endl;
			return false;
		}
	}
	else
	{
		if(image_data_type!=CV_8S)
		{
			cout<<"error: -> in fastUcharImageIntegration :输入图像数据类型不正确，应该为char型数据"<<endl;
			return false;
		}
	}

	//2.迭代计算积分图像
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

/*	一般多通道图像的快速完整积分	
	输入：
	cv::Mat& image_in		输入图像（任意类型）
	输出：
	cv::Mat& image_out		输出图像（float型）
	当函数返回false时，程序出错*/
bool fastNormalImageIntegration(cv::Mat& image_in,cv::Mat& image_out)
{
	//1.检测输入图像尺寸、类型等参数
	int image_rows=image_in.rows;
	int image_cols=image_in.cols;

	int image_channels=image_in.channels();	//图像通道数
	int image_data_type=image_in.depth();	//获取输入图像的数据类型

	if(image_data_type!=CV_32F)
	{
		image_in.convertTo(image_in,CV_32F);
		return false;
	}

	//2.迭代计算积分图像
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

/*	图像积分运算测试函数	*/
bool imageIntegrationTest()
{
	cout<<endl<<"图像积分运算测试:"<<endl;

	cv::Mat image_in,image_out;
	image_in=cv::imread("测试图片/Lena.jpg");
	cv::imshow("original image",image_in);

	double t_start=(double)cv::getTickCount();
	bool result=fastUcharImageIntegration(image_in,0,image_out);
	double t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();;
	cout<<"uchar积分用时:"<<t_used<<"s"<<endl;
	
	cv::imshow("uchar",image_out/256);

	t_start=(double)cv::getTickCount();
	result=fastNormalImageIntegration(image_in,image_out);
	t_used = ((double)cv::getTickCount()-t_start)/cv::getTickFrequency();;
	cout<<"一般积分用时:"<<t_used<<"s"<<endl;
	
	cv::imshow("normal",image_out/256);

	cv::waitKey(100000);

	return true;
}
