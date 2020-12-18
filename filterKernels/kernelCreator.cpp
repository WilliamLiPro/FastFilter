/***************************************************************
	>名称：滤波核生成函数集
	>作者：李维鹏
	>联系方式：williamli_pro@163.com
	>生成实验用，不同尺寸的各类滤波核（包括人脸模板）
	>要点：
	>1.生成高斯核
	>2.生成sobel核
	>3.生成DOG核
	>4.生成标准人脸模板

****************************************************************/

#include "kernelCreator.h"


/*	高斯核生成函数
	输入：
	int nx					滤波核x尺寸
	int ny					滤波核y尺寸
	float std_x			滤波核x标准差
	float std_y			滤波核y标准差
	输出：
	cv::Mat& out_kernel		高斯滤波核
	*/
void gaussKernelCreator(cv::Mat& out_kernel,int nx,int ny,float std_x,float std_y)
{
	cv::Mat x_kernel,y_kernel;
	x_kernel=cv::getGaussianKernel(nx,std_x,CV_32F);
	y_kernel=cv::getGaussianKernel(ny,std_y,CV_32F);

	out_kernel=y_kernel*x_kernel.t();
}

/*	sobel核生成函数(默认差分方向为y)
	输入：
	int nk					滤波核尺寸
	float std_x			高斯滤波核x标准差
	float dy				sobel核y差分
	输出：
	cv::Mat& out_kernel		高斯滤波核
	*/
void sobelKernelCreator(cv::Mat& out_kernel,int nk,float std_x,float dy)
{
	//计算高斯核
	cv::Mat x_kernel;
	x_kernel=cv::getGaussianKernel(nk,std_x,CV_32F);

	//sobel
	out_kernel.create(nk,nk,CV_32FC1);
	for(int i=0;i<nk;i++)
	{
		int center_d=i-nk/2;
		if(center_d<0)
			center_d=-(nk/2+1-abs(center_d));
		else if(center_d>0)
			center_d=nk/2+1-abs(center_d);

		out_kernel.row(i)=center_d*dy*x_kernel.t();
	}
}

/*	DOG核生成函数
	输入：
	int nk					滤波核尺寸
	float std_1			滤波核x标准差
	float std_2			滤波核y标准差
	输出：
	cv::Mat& out_kernel		高斯滤波核
	*/
void theDOGKernelCreator(cv::Mat& out_kernel,int nk,float std_1,float std_2)
{
	//计算一维DOG核
	cv::Mat vec_kernel=cv::getGaussianKernel(nk,std_1,CV_32F)-cv::getGaussianKernel(nk,std_2,CV_32F);
	float* p_vec=(float*)vec_kernel.data;

	float vec_square=0;
	for(int i=0;i<vec_kernel.rows;i++)
	{
		vec_square+=p_vec[i]*p_vec[i];
	}

	//计算2维DOG核
	out_kernel=vec_kernel*vec_kernel.t()/vec_square;
}

/*	生成标准人脸模板
	输入：
	cv::Mat& pic_in			输入图像
	int width				输出宽度
	int height				输出高度
	输出：
	cv::Mat& out_mask		输出模板
	*/
void standardFaceCreator(cv::Mat& pic_in,cv::Mat& out_mask,int width,int height)
{
	cv::Mat middle;
	cv::resize(pic_in,middle,cv::Size(width,height));

	if(middle.channels()==1)
		middle.convertTo(out_mask,CV_32F);
	else
		cv::cvtColor(middle,out_mask,CV_BGR2GRAY);
	
	out_mask.convertTo(out_mask,CV_32F);
}


void kernelCreatorTest()
{
	//1.生成高斯核
	cv::Mat out_kernel;
	gaussKernelCreator(out_kernel,5,5,3,3);
	cout<<"高斯核"<<endl<<out_kernel<<endl;

	//2.生成sobel核
	sobelKernelCreator(out_kernel,5,3,0.2);
	cout<<"sobel核"<<endl<<out_kernel<<endl;

	//3.生成DOG核
	theDOGKernelCreator(out_kernel,5,2,3);
	cout<<"DOG核"<<endl<<out_kernel<<endl;

	//4.生成标准人脸模板
	cv::Mat image_in;
	if(_waccess(L"test images/Eigenface for recognition主元.jpg",0)==0)
		image_in=cv::imread("test images/Eigenface for recognition主元.jpg");
	else if(_waccess(L"../test images/Eigenface for recognition主元.jpg",0)==0)
		image_in=cv::imread("../test images/Eigenface for recognition主元.jpg");
	else
		image_in=cv::imread("../../test images/Eigenface for recognition主元.jpg");

	standardFaceCreator(image_in,out_kernel,100,100);

	cv::imshow("人脸模板",out_kernel/255);
	cv::waitKey(10000);

}

//test
/*void main()
{
	kernelCreatorTest();
}*/