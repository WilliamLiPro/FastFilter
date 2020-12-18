/***************************************************************
	>名称：测试主程序
	>作者：李维鹏
	>联系方式：248636779@163.com
	>测试滤波核生成、核分解、快速滤波等程序
	>要点：
	>1.滤波核生成
	>2.核分解
	>3.快速滤波等程序

****************************************************************/

#include "imageIntegration.h"
#include "FastImageMaskFilter.h"

void main()
{
	string img_path = "test images/Lena.jpg";	//change into your image here
	bool re=fastImageFilterTest(img_path.c_str());
}
