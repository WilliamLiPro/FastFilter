/***************************************************************
	>���ƣ�����������
	>���ߣ���ά��
	>��ϵ��ʽ��248636779@163.com
	>�����˲������ɡ��˷ֽ⡢�����˲��ȳ���
	>Ҫ�㣺
	>1.�˲�������
	>2.�˷ֽ�
	>3.�����˲��ȳ���

****************************************************************/

#include "imageIntegration.h"
#include "FastImageMaskFilter.h"

void main()
{
	string img_path = "test images/Lena.jpg";	//change into your image here
	bool re=fastImageFilterTest(img_path.c_str());
}
