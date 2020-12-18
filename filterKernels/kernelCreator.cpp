/***************************************************************
	>���ƣ��˲������ɺ�����
	>���ߣ���ά��
	>��ϵ��ʽ��williamli_pro@163.com
	>����ʵ���ã���ͬ�ߴ�ĸ����˲��ˣ���������ģ�壩
	>Ҫ�㣺
	>1.���ɸ�˹��
	>2.����sobel��
	>3.����DOG��
	>4.���ɱ�׼����ģ��

****************************************************************/

#include "kernelCreator.h"


/*	��˹�����ɺ���
	���룺
	int nx					�˲���x�ߴ�
	int ny					�˲���y�ߴ�
	float std_x			�˲���x��׼��
	float std_y			�˲���y��׼��
	�����
	cv::Mat& out_kernel		��˹�˲���
	*/
void gaussKernelCreator(cv::Mat& out_kernel,int nx,int ny,float std_x,float std_y)
{
	cv::Mat x_kernel,y_kernel;
	x_kernel=cv::getGaussianKernel(nx,std_x,CV_32F);
	y_kernel=cv::getGaussianKernel(ny,std_y,CV_32F);

	out_kernel=y_kernel*x_kernel.t();
}

/*	sobel�����ɺ���(Ĭ�ϲ�ַ���Ϊy)
	���룺
	int nk					�˲��˳ߴ�
	float std_x			��˹�˲���x��׼��
	float dy				sobel��y���
	�����
	cv::Mat& out_kernel		��˹�˲���
	*/
void sobelKernelCreator(cv::Mat& out_kernel,int nk,float std_x,float dy)
{
	//�����˹��
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

/*	DOG�����ɺ���
	���룺
	int nk					�˲��˳ߴ�
	float std_1			�˲���x��׼��
	float std_2			�˲���y��׼��
	�����
	cv::Mat& out_kernel		��˹�˲���
	*/
void theDOGKernelCreator(cv::Mat& out_kernel,int nk,float std_1,float std_2)
{
	//����һάDOG��
	cv::Mat vec_kernel=cv::getGaussianKernel(nk,std_1,CV_32F)-cv::getGaussianKernel(nk,std_2,CV_32F);
	float* p_vec=(float*)vec_kernel.data;

	float vec_square=0;
	for(int i=0;i<vec_kernel.rows;i++)
	{
		vec_square+=p_vec[i]*p_vec[i];
	}

	//����2άDOG��
	out_kernel=vec_kernel*vec_kernel.t()/vec_square;
}

/*	���ɱ�׼����ģ��
	���룺
	cv::Mat& pic_in			����ͼ��
	int width				������
	int height				����߶�
	�����
	cv::Mat& out_mask		���ģ��
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
	//1.���ɸ�˹��
	cv::Mat out_kernel;
	gaussKernelCreator(out_kernel,5,5,3,3);
	cout<<"��˹��"<<endl<<out_kernel<<endl;

	//2.����sobel��
	sobelKernelCreator(out_kernel,5,3,0.2);
	cout<<"sobel��"<<endl<<out_kernel<<endl;

	//3.����DOG��
	theDOGKernelCreator(out_kernel,5,2,3);
	cout<<"DOG��"<<endl<<out_kernel<<endl;

	//4.���ɱ�׼����ģ��
	cv::Mat image_in;
	if(_waccess(L"test images/Eigenface for recognition��Ԫ.jpg",0)==0)
		image_in=cv::imread("test images/Eigenface for recognition��Ԫ.jpg");
	else if(_waccess(L"../test images/Eigenface for recognition��Ԫ.jpg",0)==0)
		image_in=cv::imread("../test images/Eigenface for recognition��Ԫ.jpg");
	else
		image_in=cv::imread("../../test images/Eigenface for recognition��Ԫ.jpg");

	standardFaceCreator(image_in,out_kernel,100,100);

	cv::imshow("����ģ��",out_kernel/255);
	cv::waitKey(10000);

}

//test
/*void main()
{
	kernelCreatorTest();
}*/