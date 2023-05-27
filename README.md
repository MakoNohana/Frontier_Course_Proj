# 图像处理工具说明文档

## 基于 OpenCV 、 Qt 库实现的图像处理软件


## 1. 系统介绍

### 1.1 编程语言

C++

### 1.2 开发软件环境

Windows 10、Visual Studio 2022 、Qt Creator

### 1.3 运行平台和支撑环境

OpenCV、Qt6、libjpeg、libpng

### 1.4 开发目的

对jpeg及PNG图像进行自适应双边滤波，并验证、评价不同图像的滤波、变换效果

加深对数字图像处理的理解，熟练使用 OpenCV 的各种图像处理功能。学会调用OpenCV库，并实现视频推流、实时图像识别等有趣的内容

### 1.5 主要功能

本图像处理软件的基本功能如下：

打开图像文件、显示图像、存储处理后图像，对图像添加高斯噪声、椒盐噪声、双边滤波、3×3 中值滤波、基于libjpeg编码解码、边缘检测、直方图计算与显示。

视频图像处理功能，可以对每帧视频帧进行灰度化并保存，同时基于灰度图像进行光流估计。

检测视频中人脸面部并覆盖标记，采用 OpenCV 中 harrcascade 检测框架检测人脸，并实时覆盖，可调用计算机自带的摄像头功能。

## 2. 系统分析与设计

### 2.1 需求分析

实现主要功能，打开图像文件、显示图像、存储处理后图像，对图像添加高斯噪声、椒盐噪声、双边滤波、3×3 中值滤波、基于libjpeg编码解码、边缘检测、直方图计算与显示。

视频图像处理功能，可以对每帧视频帧进行灰度化并保存，同时基于灰度图像进行光流估计。

检测视频中人脸面部并覆盖标记，采用 OpenCV 中 harrcascade 检测框架检测人脸，并实时覆盖，可调用计算机自带的摄像头功能。

暂未完成图像自适应和鼠标控制的缩放。

### 2.2 概要设计

菜单栏部分：包含的框体有文件、图像处理、视频处理、工具、帮助、关于。

每个菜单（模块）对应的功能：

- 文件：包括打开文件，保存文件，关闭文件以及退出。
- 图像处理：包括高斯噪声，椒盐噪声，解码编码，双边滤波，边缘检测，直方图计算与显示。
- 视频处理：包括灰度化与光流估计，灰度化，人脸识别。
- 工具：撤销处理，重置图片到未处理状态。
- 帮助：说明文档，关于本软件信息。
- 关于：联系作者，关于本人信息。
- 工具栏部分：包含以上功能的对于按钮，并有按钮图标。
- 图像显示部分：左面显示原图像，右面显示处理后的图像。
- TODO：左右侧图像实现鼠标控制的缩放。

### 2.3 详细设计

首先窗口界面的实现运用了 Qt Creator 进行页面搭建，随后采用 Visual Studio 2022 在 main.cpp 中进行代码编写，实现主函数运行后窗口的显示。

其次是在 Momoka.cpp 文件中编写主窗口中每个功能所对应的触发事件，包括菜单栏，工具栏按钮的对应槽函数。

最后编写对应的每个功能的实现函数，采用 OpenCV 库中封装的方法，直接对读取的图像和视频进行处理与保存，实现起来非常方便，运行起来十分高效。

额外运用了 OpenCV 中 harrcascade 检测框架，检测视频中人脸面部，并实时覆盖标记。

### 2.4 函数具体实现及其原理

#### 2.4.1 高斯噪声

高斯噪声是一种常见的随机噪声，其原理基于高斯分布（也称为正态分布）。高斯分布是一种连续概率分布，其特点是均值为 $\mu$（mu）且标准差为 $\sigma$（sigma）。高斯噪声的产生是通过在信号中添加服从高斯分布的随机值来模拟真实世界中的随机干扰。

高斯噪声的概率密度函数（PDF）可用以下公式表示：

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

其中，$x$ 表示噪声的取值，$\mu$ 是均值，$\sigma$ 是标准差。这个公式描述了在给定均值和标准差下，随机变量的概率分布情况。

在数字图像处理中，高斯噪声常用于模拟图像中的噪声，例如由于传感器噪声、通信噪声或其他环境因素引起的噪声。通过添加服从高斯分布的随机值到图像的每个像素，可以模拟这种噪声的影响。

在这段代码中，我们通过调用random库来实现高斯噪声的添加，为了可视化的展现噪声的强度，这里引入了一个窗体来修改参数实现上述公式参数的调节，具体原理如下：

调节高斯噪声的强度可以通过调整概率密度函数中的参数来实现。主要影响噪声强度的参数是均值（μ）和标准差（σ）。

- 均值（μ）：均值确定了噪声分布的中心位置。增大均值会使噪声整体偏离原始信号的平均值，因此增加了噪声的强度。减小均值会使噪声更接近于原始信号的平均值，从而降低噪声强度。

- 标准差（σ）：标准差决定了噪声分布的分散程度。增大标准差会使噪声值更分散，噪声强度也会增加。减小标准差会使噪声值更集中，从而降低噪声强度。

因此，通过增大均值和标准差可以增加高斯噪声的强度，而减小均值和标准差可以降低噪声强度。


```cpp
// 高斯噪声
void Momoka::guass_noise()
{
    ui->label_showProcessImage->clear();

    if (image.data)
    {
        // 克隆原始图像
        Mat noisyImage = image.clone();

        int noiseIntensity = 0;

        bool ok;
        noiseIntensity = QInputDialog::getInt(this, "设置高斯噪声强度", "请输入噪声强度（0-255）:", 20, 0, 255, 1, &ok);
        if (!ok)
            return;

        // 添加高斯噪声
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 1.0); // 均值为0，标准差为1的正态分布
        for (int i = 0; i < noisyImage.rows; ++i) {
            for (int j = 0; j < noisyImage.cols; ++j) {
                for (int k = 0; k < noisyImage.channels(); ++k) {
                    double noise = distribution(generator) * noiseIntensity; // 控制噪声的强度
                    double pixelValue = noisyImage.at<Vec3b>(i, j)[k] + noise;
                    noisyImage.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(pixelValue);
                }
            }
        }

        // 通过 label 方式显示图片
        display_MatInQT(ui->label_showOriginalImage, image);
        display_MatInQT(ui->label_showProcessImage, noisyImage);
    }
    else
    {
        QIcon* icon = new QIcon(".\\Resource Files\\Momoka.ico");
        QMessageBox message(QMessageBox::Information, tr("提示"), tr("处理图片失败！"));
        message.setWindowIcon(*icon);
        QPushButton* okbutton = (message.addButton(tr("确定"), QMessageBox::AcceptRole));
        message.exec();
    }
}
```

#### 2.4.2 椒盐噪声

椒盐噪声是一种常见的图像噪声，它的原理是在图像中随机出现黑色和白色像素点，模拟噪声干扰引起的突发性亮度变化。椒盐噪声常常由于传感器故障、数据传输错误或图像采集过程中的干扰引起。

椒盐噪声可以使用以下概率密度函数表示：

$$
p(x) = 
\begin{cases}
P_s & \text{if } x = x_s ,
\
P_p & \text{if } x = x_p \

0 & \text{otherwise}
\end{cases}
$$

其中，$x$ 表示图像像素值，$x_s$ 和 $x_p$ 分别表示椒噪声和盐噪声的像素值。$P_s$ 和 $P_p$ 分别是椒噪声和盐噪声的概率，通常可以控制它们的比例来调节噪声强度。

在上述公式中，如果像素值等于椒噪声像素值 $x_s$，则概率密度函数的值为 $P_s$；如果像素值等于盐噪声像素值 $x_p$，则概率密度函数的值为 $P_p$。否则，概率密度函数的值为零，表示其他像素值的概率为零。

通过调节椒噪声和盐噪声的概率，可以控制椒盐噪声的强度和比例，以适应不同的应用需求,这里和添加高斯噪声一样，提供了一个子窗体给用户调节噪声的强度。

```cpp
void Momoka::impulse_noise()
{
    ui->label_showProcessImage->clear();

    if (image.data)
    {
        // 获取噪声强度数据
        int noisePercent = 0;

        // 通过 QInputDialog 输入噪声强度
        bool ok;
        noisePercent = QInputDialog::getInt(this, "提示", "请输入椒盐噪声强度：（范围为 0 ~ 100 之间）", 50, 0, 100, 1, &ok);

        if (ok)
        {
            // 添加椒盐噪声
            Mat noisyImage = image.clone();

            int numPixels = image.rows * image.cols;
            int numNoisePixels = numPixels * noisePercent / 100;

            std::default_random_engine generator;
            std::uniform_int_distribution<int> distribution(0, numPixels - 1);

            for (int i = 0; i < numNoisePixels; ++i)
            {
                int row = distribution(generator) % image.rows;
                int col = distribution(generator) % image.cols;

                if (distribution(generator) % 2 == 0)
                {
                    // 添加椒噪声（变为黑色）
                    noisyImage.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
                }
                else
                {
                    // 添加盐噪声（变为白色）
                    noisyImage.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
                }
            }

            // 通过 label 方式显示图片
            display_MatInQT(ui->label_showOriginalImage, image);
            display_MatInQT(ui->label_showProcessImage, noisyImage);
        }
    }
    else
    {
        QIcon* icon = new QIcon(".\\Resource Files\\Momoka.ico");
        QMessageBox message(QMessageBox::Information, tr("提示"), tr("处理图片失败！"));
        message.setWindowIcon(*icon);
        QPushButton* okbutton = (message.addButton(tr("确定"), QMessageBox::AcceptRole));
        message.exec();
    }
}
```

#### 2.4.3 解码编码及其原理

这段代码是一个图像解码和编码的过程。它使用libjpeg库对图像进行解码，然后将解码后的图像数据存储到Mat对象中，并通过label在界面上显示原始图像和解码后的图像。

代码的主要步骤如下：

- 初始化libjpeg解码结构和错误处理结构。
- 设置JPEG解码数据源为图像数据。
- 开始解码，读取图像的头部信息。
- 分配内存空间用于存储解码后的图像数据。
- 创建一个Mat对象来保存解码后的图像，指定图像的大小和通道数。
- 逐行解码图像数据，并将解码后的数据复制到Mat对象中的每一行。
- 完成解码过程，释放相关资源。
- 使用label在界面上显示原始图像和解码后的图像。
- 如果图像数据存在，代码会执行解码过程并将解码后的图像显示在界面上。如果图像数据为空，说明处理图片失败，会弹出一个提示窗口显示处理失败的信息。

在安装libjpeg库的时候，请先安装CMake的正确版本，或者通过Visual Studio（UMake）进行构建，注意修改其中一些头文件或依赖为你本机上的位置。

```cpp
// 视频灰度化
//解码编码
void Momoka::decode_encode()
{
    ui->label_showProcessImage->clear();

    if (image.data)
    {
        // 使用libjpeg进行图像解码
        struct jpeg_decompress_struct cinfo;
        struct jpeg_error_mgr jerr;

        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_decompress(&cinfo);

        // 设置JPEG解码数据源
        jpeg_mem_src(&cinfo, image.data, image.size().width * image.size().height * image.channels());

        // 开始解码
        jpeg_read_header(&cinfo, TRUE);
        jpeg_start_decompress(&cinfo);

        // 分配解码后的图像数据存储空间
        int row_stride = cinfo.output_width * cinfo.output_components;
        JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

        // 创建解码后的图像Mat对象
        Mat decodedImage(cinfo.output_height, cinfo.output_width, CV_8UC3);

        // 逐行解码并保存到Mat中
        int row = 0;
        while (cinfo.output_scanline < cinfo.output_height)
        {
            jpeg_read_scanlines(&cinfo, buffer, 1);
            memcpy(decodedImage.ptr(row), buffer[0], row_stride);
            row++;
        }

        // 结束解码
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);

        // 通过 label 方式显示图片
        display_MatInQT(ui->label_showOriginalImage, image);
        display_MatInQT(ui->label_showProcessImage, decodedImage);
    }
    else
    {
        QIcon* icon = new QIcon(".\\Resource Files\\Momoka.ico");
        QMessageBox message(QMessageBox::Information, tr("提示"), tr("处理图片失败！"));
        message.setWindowIcon(*icon);
        QPushButton* okbutton = (message.addButton(tr("确定"), QMessageBox::AcceptRole));
        message.exec();
    }
}
```

#### 2.4.4 光流估计及其原理

光流估计是计算机视觉中的一个重要任务，用于估计连续帧之间的像素运动信息。光流指的是图像中每个像素点在时间上的位移量，描述了物体在图像中的运动轨迹。

光流估计基于以下假设：
1. 亮度恒定假设：在短时间内，相邻像素的亮度值保持不变，即在连续帧之间的像素强度保持不变。
2. 空间平滑假设：相邻像素具有相似的运动。

根据这些假设，光流估计可以通过以下步骤进行：
1. 特征提取：选择图像中的特征点，通常使用角点检测算法来提取具有较大灰度变化的像素点。
2. 特征跟踪：在相邻帧之间追踪选定的特征点，通过匹配特征点的像素值来计算它们之间的位移向量。
3. 光流计算：基于特征点的位移向量，通过光流算法估计其他像素点的光流。

在光流估计中，常用的光流方程为：

$$
\
I_xu + I_yv + I_t = 0
\
$$

其中，$I_x$和$I_y$是图像灰度关于空间坐标的梯度，$I_t$是图像灰度关于时间的变化率，$u$和$v$分别表示像素在x和y方向上的位移。

- Horn-Schunck算法

Horn-Schunck算法是一种经典的光流估计方法，基于以下能量函数进行优化：

$$
E = \int \left((I_xu + I_yv + I_t)^2 + \alpha^2 (|\nabla u|^2 + |\nabla v|^2)\right) dxdy
$$

其中，$\nabla u$和$\nabla v$分别表示位移向量$u$和$v$的梯度，$\alpha$是一个正则化参数。

通过最小化能量函数，可以得到优化后的位移场，进而估计图像的光流。

- 证明

这里给出光流方程的简要证明：

假设在时间 $t\$和 $t + \Delta t$ 之间，一个像素在 $x$ 和 $y$ 方向上的位移分别为 $u$ 和 $v$。根据亮度恒定假设，我们可以得到:

$$
I(x, y, t) = I(x + u, y + v, t + \Delta t)
$$

对上式进行泰勒展开，忽略高阶项，我们得到：

$$
I(x, y, t) + I_x u + I_y v + I_t = I(x, y, t + \Delta t)
$$

整理后得到光流方程：

$$
I_xu + I_yv + I_t = 0
$$

因此，光流方程描述了像素在时间上的位移与像素梯度之间的关系。

这些都能够在OpenCV强大的库函数里面调用和实现。鉴于光流这部分理论知识较多，视频推流的具体操作在“灰度视频”里会详细叙述。

```cpp
void Momoka::optical_flow_estimation()
{
    QString filename =
        QFileDialog::getOpenFileName(this, tr("打开视频"),
            ".",
            tr("Video file(*.mp4 *.avi)"));

    String fileName = filename.toStdString();

    QString savePath;
    String fileOut;

    if ((!filename.isNull()) || (!filename.isEmpty()))
    {
        savePath = QFileDialog::getSaveFileName(0, "请选择视频保存路径", ".\\Video Files\\recorded.mp4", "mp4(*.mp4);;avi(*.avi);;所有文件(*.*)");
    }

    if ((!savePath.isNull()) || (!savePath.isEmpty()))
    {
        fileOut = savePath.toStdString();
    }
    else
    {
        return;
    }

    VideoCapture inVid(fileName);

    Mat prevFrame, currFrame, prevGray, currGray;
    const char winIn[] = "Input Video", winOut[] = "Optical Flow";
    double fps = 30;    // 每秒的帧数

    // 打开视频文件，具体文件名需先查看
    if (!inVid.isOpened())
    {
        cout << "无法打开视频文件！\n";
        return;
    }

    // 获取输入视频的宽度和高度
    int width = (int)inVid.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)inVid.get(CAP_PROP_FRAME_HEIGHT);
    cout << "width = " << width << ", height = " << height << endl;

    VideoWriter recVid(fileOut, VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(width, height), 0);

    if (!recVid.isOpened())
    {
        cout << "无法创建输出视频文件！\n";
        return;
    }

    // 创建两个窗口，用于显示输入视频和光流估计结果
    //namedWindow(winIn);
    namedWindow(winOut);

    // 读取第一帧
    inVid >> prevFrame;
    cvtColor(prevFrame, prevGray, COLOR_BGR2GRAY);

    while (true)
    {
        // 读取下一帧
        inVid >> currFrame;
        if (currFrame.empty())
            break;

        // 将当前帧转换为灰度
        cvtColor(currFrame, currGray, COLOR_BGR2GRAY);

        // 计算光流
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prevGray, currGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // 可视化光流
        cv::Mat flowVis;
        cv::cvtColor(prevFrame, flowVis, cv::COLOR_BGR2GRAY);
        cv::cvtColor(flowVis, flowVis, cv::COLOR_GRAY2BGR);

        // 将光流可视化为彩色箭头
        float stepSize = 16; // 箭头的间隔
        for (int y = 0; y < height; y += stepSize)
        {
            for (int x = 0; x < width; x += stepSize)
            {
                // 获取光流向量
                cv::Point2f flowVec = flow.at<cv::Point2f>(y, x);

                // 绘制箭头
                cv::arrowedLine(flowVis, cv::Point(x, y), cv::Point(x + flowVec.x, y + flowVec.y), cv::Scalar(0, 255, 0), 2);
            }
        }

        // 在光流窗口中显示光流结果
        //cv::imshow(winIn, flow);
        cv::imshow(winOut, flowVis);

        // 将当前帧写入输出视频文件
        recVid << flowVis;

        if (waitKey(1000 / fps) >= 0)
        {
            break;
        }

        // 更新前一帧和前一帧的灰度图像
        prevFrame = currFrame.clone();
        prevGray = currGray.clone();
    }

    inVid.release();    // 关闭视频文件

    QIcon* icon = new QIcon(".\\Resource Files\\Momoka.ico");
    QMessageBox message(QMessageBox::Information, tr("提示"), tr("保存视频成功！\n保存路径：.\\Video Files\\recorded.mp4"));
    message.setWindowIcon(*icon);
    QPushButton* okbutton = (message.addButton(tr("确定"), QMessageBox::AcceptRole));
    message.exec();
}
```

#### 2.4.5 双边滤波及其原理

首先在代码实现这部分，依然是和噪声一样引入了用户可控的调节器，让用户可以体会到滤波强度带来的变化并且判定滤波的效果。鉴于双边滤波本身效果不明显，我们通过修改参数让它的效果放大了：
```cpp
bilateralFilter(image, bilateralFiltered, 0, filterStrength * 10, 15);
//这里参数乘了10
```
下面简单介绍一下原理

双边滤波是一种常用的图像滤波方法，它可以在保持边缘信息的同时对图像进行平滑处理。双边滤波考虑了像素的空间距离和像素的灰度相似性，从而能够有效地去除噪声并保留图像的细节。

- 原理

双边滤波基于以下两个关键概念：
1. 空间距离权重：像素之间的空间距离越近，它们的权重越大，说明它们在图像中更接近。
2. 灰度相似性权重：像素之间的灰度值越接近，它们的权重越大，说明它们在图像中更相似。

通过将空间距离权重和灰度相似性权重结合起来，双边滤波可以计算出每个像素的加权平均值，从而实现滤波效果。

- 数学公式

双边滤波可以用以下数学公式表示：

$$
I_{\text{filtered}}(x, y) = \frac{1}{W_p} \sum_{i=-k}^{k} \sum_{j=-k}^{k} w(i, j) \cdot I(x+i, y+j)
$$

其中，$I_{\text{filtered}}(x, y)$表示滤波后的图像中的像素值，$I(x+i, y+j)$表示原始图像中的像素值，$w(i, j)$表示像素之间的权重，$W_p$表示权重的归一化因子。

双边滤波的权重计算公式为：

$$
w(i, j) = w_s(i, j) \cdot w_r(I(x+i, y+j) - I(x, y))
$$

其中，$w_s(i, j)$表示空间距离权重，$w_r$表示灰度相似性权重。

### 证明

双边滤波的原理可以通过以下证明进行解释。

假设我们有一个窗口，在窗口内的像素与中心像素的空间距离为$d$，灰度值差为$r$。我们希望通过双边滤波来计算中心像素的滤波值。

首先，我们定义空间距离权重$w_s(i, j)$：

$$
w_s(i, j) = \exp\left(-\frac{i^2+j^2}{2\sigma_s^2}\right)
$$

其中，$\sigma_s$是空间距离的标准差。

接下来，我们定义灰度相似性权重$w_r$：

$$
w_r(r) = \exp\left(-\frac{r^2}{2\sigma_r^2}\right)
$$

其中，$\sigma_r$是灰度相似性的标准差。

根据双边滤波的权重计算公式：

$$
w(i, j) = w_s(i, j) \cdot w_r(I(x+i, y+j) - I(x, y))
$$

我们可以将滤波后的像素值$I_{\text{filtered}}(x, y)$表示为：

$$
I_{\text{filtered}}(x, y) = \frac{1}{W_p} \sum_{i=-k}^{k} \sum_{j=-k}^{k} w(i, j) \cdot I(x+i, y+j)
$$

通过将权重代入，我们可以得到：

$$
I_{\text{filtered}}(x, y) = \frac{1}{W_p} \sum_{i=-k}^{k} \sum_{j=-k}^{k} w_s(i, j) \cdot w_r(I(x+i, y+j) - I(x, y)) \cdot I(x+i, y+j)
$$

继续化简，我们可以得到：

$$
I_{\text{filtered}}(x, y) = \frac{1}{W_p} \sum_{i=-k}^{k} \sum_{j=-k}^{k} w_s(i, j) \cdot w_r(I(x+i, y+j) - I(x, y)) \cdot I(x+i, y+j)
$$

这证明了双边滤波的原理和公式。

```cpp
//双边滤波
void Momoka::bilateral_filter()
{
    ui->label_showProcessImage->clear();

    if (image.data)
    {
        // 获取滤波强度数据
        double filterStrength = 0.0;

        // 通过 QInputDialog 输入滤波强度
        bool ok;
        filterStrength = QInputDialog::getDouble(this, "提示", "请输入滤波强度：（范围为 0.0 ~ 10.0 之间）", 10.0, 0.0, 100.0, 2, &ok);

        if (ok)
        {
            // 双边滤波
            Mat bilateralFiltered;
            bilateralFilter(image, bilateralFiltered, 0, filterStrength * 10, 15);

            // 通过 label 方式显示图片
            display_MatInQT(ui->label_showOriginalImage, image);
            display_MatInQT(ui->label_showProcessImage, bilateralFiltered);
        }
    }
    else
    {
        QIcon* icon = new QIcon(".\\Resource Files\\Momoka.ico");
        QMessageBox message(QMessageBox::Information, tr("提示"), tr("处理图片失败！"));
        message.setWindowIcon(*icon);
        QPushButton* okbutton = (message.addButton(tr("确定"), QMessageBox::AcceptRole));
        message.exec();
    }
}
```

#### 2.4.6 边缘检测及其原理

这一部分代码基本可以直接引用OpenCV库内已经成型的函数，需要写的代码量不大。做这个部分的主要目的是希望以后画画提取线稿会方便一些，同时在未来也可以做扫描仪用，如果有0.1.2版本的话准备做一个白底黑线的，同样在弹出子窗口中进行选择。

- 边缘检测

边缘检测是计算机视觉和图像处理中的一项基本任务，用于提取图像中物体的边界信息。边缘通常表示图像中灰度或颜色变化的区域。

- 原理

边缘检测的原理基于以下观察：
1. 图像中的边缘通常具有较大的梯度。
2. 边缘在图像中的位置通常与像素灰度或颜色变化的位置相对应。

基于这些观察，边缘检测可以通过以下步骤进行：
1. 图像平滑：使用滤波器（如高斯滤波器）对图像进行平滑处理，以减少噪声。
2. 梯度计算：计算图像的梯度，通常使用Sobel、Prewitt等算子。
3. 非极大值抑制：对梯度幅值进行局部最大值抑制，保留梯度方向上的局部最大值点。
4. 阈值处理：根据设定的阈值，将非极大值抑制后的梯度幅值进行二值化，得到二值边缘图像。

- 数学公式

在边缘检测中，常用的数学公式包括梯度计算和非极大值抑制。

- 梯度计算

梯度计算通常使用Sobel算子，其中$G_x$和$G_y$表示图像在水平和垂直方向上的梯度：

$$
G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I
$$

$$
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I
$$

其中，$*$表示卷积操作，$I$表示输入图像。

- 非极大值抑制

非极大值抑制用于提取边缘的细化线条。对于梯度图像，取梯度方向上的局部最大值点，将其它位置的梯度值置为0。

- 证明

这里给出非极大值抑制的简要证明：

假设有一个灰度图像 $I(x, y)$ 和其在水平方向上的梯度 $G_x(x, y)$ 和垂直方向上的梯度 $G_y(x, y)$。我们希望对梯度图像进行非极大值抑制，提取出边缘的细化线条。

非极大值抑制的思想是，在每个像素位置上，通过比较该位置上的梯度幅值和沿着梯度方向的两个邻域像素的梯度幅值，选择局部最大值作为非极大值抑制后的梯度值。

具体步骤如下：
1. 对每个像素位置 $(x, y)$，获取其梯度幅值 $M(x, y)$ 和梯度方向 $\theta(x, y)$。
2. 将梯度方向 $\theta(x, y)$ 量化为离散的四个方向之一：0°、45°、90°和135°。
3. 根据梯度方向，选择相应的邻域像素进行比较。例如，对于0°的梯度方向，比较 $(x, y)$ 位置的梯度幅值与其水平方向上的两个邻域像素的梯度幅值。
4. 如果 $(x, y)$ 位置的梯度幅值大于邻域像素的梯度幅值，则保留该梯度值；否则，将梯度值置为0。
5. 重复步骤2至步骤4，对每个像素位置进行非极大值抑制。

经过非极大值抑制后，我们可以得到细化的边缘线条，提取出图像中的边缘信息。



```cpp
void Momoka::edge_detection()
{
    ui->label_showProcessImage->clear();

    if (image.data)
    {
        // 边缘检测
        Canny(image, edgeDetection, 150, 100, 3);
        // 通过 label 方式显示图片
        display_MatInQT(ui->label_showOriginalImage, image);
        display_MatInQT(ui->label_showProcessImage, edgeDetection);
    }
    else
    {
        QIcon* icon = new QIcon(".\\Resource Files\\Momoka.ico");
        QMessageBox message(QMessageBox::Information, tr("提示"), tr("处理图片失败！"));
        message.setWindowIcon(*icon);
        QPushButton* okbutton = (message.addButton(tr("确定"), QMessageBox::AcceptRole));
        message.exec();
    }
}
```

#### 2.4.7 直方图及其原理

直方图是图像处理中常用的工具，用于描述图像中像素值的分布情况。在图像识别任务中，直方图常用于表示图像的特征，提供有关图像内容的统计信息。

- 原理

图像识别直方图的原理基于以下假设：
1. 图像的像素值反映了图像的特征信息。
2. 图像中不同特征的像素值分布具有差异。

根据这些假设，图像识别直方图可以通过以下步骤计算：
1. 选择感兴趣的图像区域。
2. 将该区域划分为若干个像素值范围（通常是离散的）。
3. 统计每个像素值范围内像素的数量或占比，得到直方图。

- 数学公式

图像识别直方图的数学公式如下所示：

$$
H(i) = \sum_{x, y} \delta(I(x, y) - i)
$$

其中，$H(i)$表示第$i$个像素值范围内的像素数量或占比，$I(x, y)$表示图像在位置$(x, y)$处的像素值，$\delta(\cdot)$是指示函数，当括号内的条件为真时取值为1，否则取值为0。

- 证明

我们假设图像中存在一种特征，其像素值范围为$i$到$i+\Delta i$。我们希望证明直方图中对应的$H(i)$能够准确反映该特征的像素数量或占比。

考虑图像中的每个像素点$(x, y)$，它的像素值为$I(x, y)$。根据定义，当且仅当$I(x, y)$位于范围$i$到$i+\Delta i$内时，$\delta(I(x, y) - i)$取值为1，否则取值为0。

因此，$H(i)$可以表示为所有满足条件$I(x, y) \in [i, i+\Delta i]$的像素点的数量或占比之和。通过统计满足条件的像素数量或占比，我们得到了该特征在图像中的直方图。

下面按照这个思路简单实现了一下，缺点是输出图片的尺寸适配还没做太好。
```cpp
// 直方图计算与显示
void Momoka::on_action_histogramCalculationAndDisplay_triggered()
{
    Mat src, dst;
    src = image;

    if (image.data)
    {
        // 将多通道图像分为单通道图像
        // 单通道图像 vector
        std::vector<Mat> bgr_planes;
        split(src, bgr_planes);

        // 直方图参数
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRanges = {range};
        Mat b_hist, g_hist, r_hist;

        // 求出直方图的数据
        calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize,
            &histRanges, true, false);
        calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize,
            &histRanges, true, false);
        calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize,
            &histRanges, true, false);

        // 画出直方图
        int hist_h = 400;
        int hist_w = 512;
        int bin_w = hist_w / histSize;    // 直方图的步数

        Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));

        // 将数据归一化到直方图的图像尺寸中来
        normalize(b_hist, b_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
        normalize(g_hist, g_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
        normalize(r_hist, r_hist, 0, hist_h, NORM_MINMAX, -1, Mat());

        // 0-255 的像素值，画出每个像素值的连线
        //注意：图像中的坐标是以左上角为原点向右下方延伸
        for (int i = 1; i < histSize; ++i)
        {
            line(histImage, Point((i - 1) * bin_w, hist_h - cvRound(b_hist.at<float>(i - 1))),
                Point(i * bin_w, hist_h - cvRound(b_hist.at<float>(i))), Scalar(0, 0, 255), 2, LINE_AA);
            line(histImage, Point((i - 1) * bin_w, hist_h - cvRound(g_hist.at<float>(i - 1))),
                Point(i * bin_w, hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, LINE_AA);
            line(histImage, Point((i - 1) * bin_w, hist_h - cvRound(r_hist.at<float>(i - 1))),
                Point(i * bin_w, hist_h - cvRound(r_hist.at<float>(i))), Scalar(255, 0, 0), 2, LINE_AA);
        }

        // 通过 label 方式显示图片
        display_MatInQT(ui->label_showOriginalImage, image);
        display_MatInQT(ui->label_showProcessImage, histImage);
    }
    else
    {
        QIcon* icon = new QIcon(".\\Resource Files\\Momoka.ico");
        QMessageBox message(QMessageBox::Information, tr("提示"), tr("处理图片失败！"));
        message.setWindowIcon(*icon);
        QPushButton* okbutton = (message.addButton(tr("确定"), QMessageBox::AcceptRole));
        message.exec();
    }
}
```

#### 2.4.8 灰度视频流
- 视频灰度化
该程序实现了将视频转换为灰度图像的功能。用户可以选择一个视频文件（支持格式：.mp4和.avi），程序将读取该视频文件并将每一帧转换为灰度图像，最后将转换后的帧保存为新的视频文件。

- 功能介绍
1. 打开视频文件：用户可以通过选择视频文件对话框来指定要处理的视频文件。
2. 选择视频保存路径：用户可以选择视频保存的路径和文件名，保存的视频文件格式支持.mp4和.avi。
3. 视频转换：程序将逐帧读取原始视频文件，并将每一帧转换为灰度图像。
4. 视频保存：转换后的灰度图像帧将按原始帧率保存为新的视频文件。
5. 显示原始视频和灰度视频：程序将在两个窗口中分别显示原始视频和转换后的灰度视频帧。
- 使用步骤
1. 打开程序后，点击菜单中的"视频灰度化"选项。
2. 弹出的文件对话框中选择要处理的视频文件。
3. 弹出的保存文件对话框中选择视频保存的路径和文件名。
4. 程序开始读取视频文件，并将每一帧转换为灰度图像。
5. 转换后的帧将保存为新的视频文件。
在两个窗口中分别显示原始视频和灰度视频帧。
6. 程序运行结束后，会弹出提示对话框显示保存视频成功，并给出保存路径。
- 实现过程

1. 用户界面：程序提供了一个简单的用户界面，通过菜单选项触发视频灰度化功能。
打开视频文件：用户通过选择视频文件对话框来指定要处理的视频文件路径和名称。程序使用Qt的QFileDialog类来实现文件选择对话框。

2. 文件路径处理：获取用户选择的视频文件路径，并将其转换为C++标准字符串格式（std::string）以便后续处理。
3. 选择视频保存路径：用户可以选择视频保存的路径和文件名。同样使用QFileDialog类实现文件保存对话框，获取保存路径并转换为std::string格式。
4. 打开视频：使用OpenCV的VideoCapture类打开用户选择的视频文件。如果无法打开视频文件，则程序会输出错误信息并返回。
5. 获取视频信息：通过OpenCV的函数获取输入视频的宽度和高度，以便后续创建输出视频的大小。
6. 创建输出视频：使用OpenCV的VideoWriter类创建一个新的视频文件，并设置视频编码器、帧率和大小。
7. 创建窗口：使用OpenCV的namedWindow函数创建两个窗口，用于显示原始视频和转换后的灰度视频帧。
8. 视频转换：进入主循环，使用VideoCapture的>>运算符逐帧读取视频。如果读取的帧为空，则表示视频已经读取完毕，释放资源并退出循环。
9. 帧转换：使用OpenCV的cvtColor函数将读取的帧转换为灰度图像。
10. 视频帧保存：使用VideoWriter的<<运算符将转换后的灰度帧写入输出视频文件。
11. 显示视频帧：使用OpenCV的imshow函数在窗口中显示原始视频和转换后的灰度视频帧。
12. 等待按键：使用OpenCV的waitKey函数等待用户按键，每帧显示的时间间隔由帧率决定。如果用户按下任意键，则退出循环。
13. 关闭视频：释放VideoCapture和VideoWriter的资源，关闭输入和输出视频文件。
弹出提示对话框：使用Qt的QMessageBox类弹出一个消息框，提示保存视频成功，并显示保存路径。
```cpp
// 视频灰度化
void Momoka::on_action_grayscaleVideo_triggered()
{
    QString filename =
        QFileDialog::getOpenFileName(this, tr("打开视频"),
            ".",
            tr("Video file(*.mp4 *.avi)"));

    String fileName = filename.toStdString();

    QString savePath;
    String fileOut;

    if ((!filename.isNull()) || (!filename.isEmpty()))
    {
        savePath = QFileDialog::getSaveFileName(0, "请选择视频保存路径", ".\\Video Files\\recorded.mp4", "mp4(*.mp4);;avi(*.avi);;所有文件(*.*)");
    }

    if ((!savePath.isNull()) || (!savePath.isEmpty()))
    {
        fileOut = savePath.toStdString();
    }
    else
    {
        return;
    }

    VideoCapture inVid(fileName);

    Mat inFrame, outFrame;
    const char winIn[] = "Grabbing...", winOut[] = "Recording...";
    double fps = 60;    // 每秒的帧数

    // 打开摄像头，具体设备文件需先查看，参数为 0，则为默认摄像头
    if (!inVid.isOpened())
    {    // 检查错误
        cout << "错误相机未就绪！\n";
        return;
    }

    // 获取输入视频的宽度和高度
    int width = (int)inVid.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)inVid.get(CAP_PROP_FRAME_HEIGHT);
    cout << "width = " << width << ", height = " << height << endl;

    VideoWriter recVid(fileOut, VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(width, height), 0);

    if (!recVid.isOpened())
    {
        cout << "错误视频文件未打开！\n";
        return;
    }

    // 为原始视频和最终视频创建两个窗口
    namedWindow(winIn);
    namedWindow(winOut);

    while (true)
    {
        // 从摄像机读取帧（抓取并解码）以流的形式进行
        inVid >> inFrame;

        if (inFrame.empty())
        {
            inVid.release();
            break;
        }
        // 将帧转换为灰度
        cvtColor(inFrame, outFrame, COLOR_BGR2GRAY);

        // 将帧写入视频文件（编码并保存）以流的形式进行
        recVid << outFrame;

        imshow(winIn, inFrame);    // 在窗口中显示帧
        imshow(winOut, outFrame);    // 在窗口中显示帧

        if (waitKey(1000 / fps) >= 0)
        {
            break;
        }
    }

    inVid.release();    // 关闭摄像机
    QIcon* icon = new QIcon(".\\Resource Files\\Momoka.ico");
    QMessageBox message(QMessageBox::Information, tr("提示"), tr("保存视频成功！\n保存路径：.\\Video Files\\recorded.mp4"));
    message.setWindowIcon(*icon);
    QPushButton* okbutton = (message.addButton(tr("确定"), QMessageBox::AcceptRole));
    message.exec();
}
```

#### 2.4.9 人脸识别

主要实现如下

以下是对给定的人脸识别程序的实现过程进行详细说明：

主函数部分

```cpp
// 视频灰度化
void Momoka::on_action_grayscaleVideo_triggered()
{
    // 省略部分代码...

    VideoCapture inVid(fileName);

    Mat inFrame, outFrame;
    const char winIn[] = "Grabbing...", winOut[] = "Recording...";
    double fps = 60;    // 每秒的帧数

    // 打开摄像头，具体设备文件需先查看，参数为 0，则为默认摄像头
    if (!inVid.isOpened())
    {    // 检查错误
        cout << "错误相机未就绪！\n";
        return;
    }

    // 获取输入视频的宽度和高度
    int width = (int)inVid.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)inVid.get(CAP_PROP_FRAME_HEIGHT);
    cout << "width = " << width << ", height = " << height << endl;

    VideoWriter recVid(fileOut, VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(width, height), 0);

    // 省略部分代码...

    while (true)
    {
        // 从摄像机读取帧（抓取并解码）以流的形式进行
        inVid >> inFrame;

        if (inFrame.empty())
        {
            inVid.release();
            break;
        }
        // 将帧转换为灰度
        cvtColor(inFrame, outFrame, COLOR_BGR2GRAY);

        // 将帧写入视频文件（编码并保存）以流的形式进行
        recVid << outFrame;

        // 省略部分代码...
    }

    // 省略部分代码...
}
```
cvtColor函数
```cpp
// 将彩色帧转换为灰度帧
cvtColor(inFrame, outFrame, COLOR_BGR2GRAY);
```
该函数用于将输入的彩色帧（BGR格式）转换为灰度帧。

- inFrame：输入的彩色帧
- outFrame：输出的灰度帧

VideoWriter类
```cpp
Copy code
VideoWriter recVid(fileOut, VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, Size(width, height), 0);
```
该类用于创建视频编码器，并将帧写入视频文件。

- fileOut：输出视频文件路径

- VideoWriter::fourcc('X', 'V', 'I', 'D')：指定视频编码格式为XVID

- fps：每秒的帧数

- Size(width, height)：输出视频的尺寸，与输入视频保持一致

- 0：指定视频是否为彩色帧（0表示灰度帧）

循环读取帧并处理
```cpp
while (true)
{
    // 从摄像机读取帧（抓取并解码）以流的形式进行
    inVid >> inFrame;

    // 省略部分代码...

    // 将帧转换为灰度
    cvtColor(inFrame, outFrame, COLOR_BGR2GRAY);

    // 将帧写入视频文件（编码并保存）以流的形式进行
    recVid << outFrame;

    // 省略部分代码...
}
```
该循环用于连续读取视频帧，将每一帧转换为灰度，并将灰度帧写入视频文件。

- inVid >> inFrame：从摄像机读取帧并存储在inFrame中
- cvtColor(inFrame, outFrame, COLOR_BGR2GRAY)：将彩色帧转换为灰度帧
- recVid << outFrame：将灰度帧写入视频文件

释放资源
```cpp
Copy code
inVid.release();    // 关闭摄像机
```
该语句用于释放摄像机资源，关闭摄像头。


## 3. 主要运行界面

### 3.1 主窗口

![主窗口](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/nmain.png)
_图 1 主窗口_

### 3.2 高斯噪声

![设置高斯噪声强度](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/21.png)
_图2  高斯噪声 设置粒度_

![高斯噪声](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/22.png)
_图3  高斯噪声 较低粒度_

![高斯噪声](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/4.png)
_图4  高斯噪声 较高粒度_

### 3.3 椒盐噪声

![设置椒盐噪声强度](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/5.png)
_图 5 椒盐噪声 设置粒度_

![椒盐噪声](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/6.png)
_图 6 椒盐噪声 较低强度_

![椒盐噪声](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/7.png)
_图 7 椒盐噪声 较高强度_

### 3.4 libjpeg解编码

![libjpeg解编码](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/8.png)
_图 8 libjpeg解编码_

### 3.5 双边滤波

![设置双边滤波强度](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/9.png)
_图 9 双边滤波 设置强度_

![双边滤波](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/10.png)
_图 10 双边滤波 较低强度_

![双边滤波](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/11.png)
_图 11 双边滤波 较高强度_

### 3.6 边缘检测

![边缘检测](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/12.png)
_图 12 边缘检测_

### 3.7 直方图

![直方图](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/13.png)
_图 13 直方图_

### 3.8 视频灰度＋光流

![光流](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/14.png)
_图 14 视频灰度＋光流

### 3.9视频灰度化

![视频灰度化](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/15.png)
_图 15 视频灰度化_

### 3.10 人脸识别

![人脸识别](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/16.png)

_图 16 人脸识别_

### 3.11 说明文档

![说明文档](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/17.png)
_图 17 说明文档_

### 3.12 联系作者

![联系作者](https://cdn.jsdelivr.net/gh/makonohana/cdn@master/img/18.png)
_图 18 联系作者_


## 4. 总结
在本次实验中，我们使用了Qt6和OpenCV开发了一个图像识别软件。通过这个实验，我们获得了以下经验和收获：

1. 学习了算法优化：在图像识别领域，算法的效率和准确性非常重要。我们学会了通过改进算法来提高图像识别的性能。通过尝试不同的算法和参数调整，我们能够找到更有效的方法来识别和处理图像。

2. 导入外部库和配置环境：使用OpenCV作为图像处理库，我们学会了如何导入外部库，并在Qt项目中正确配置环境，以便能够使用库中的功能。这使我们能够利用OpenCV的强大功能来处理图像，例如图像滤波、特征提取和目标检测等。

3. 使用CMake构建项目：CMake是一个跨平台的构建工具，我们学会了如何使用CMake来管理我们的项目。通过编写CMakeLists.txt文件，我们可以定义项目的结构、依赖关系和构建过程。这样可以方便地在不同平台和环境中构建和部署我们的图像识别软件。

4. 多线程编程：为了提高图像处理的速度和响应性，我们尝试了多线程编程。通过将图像处理任务分配给多个线程来并行处理，我们能够充分利用多核处理器的计算能力，提高图像处理的效率。同时，我们也学会了如何正确地管理线程之间的同步和共享资源的访问，以避免竞态条件和数据不一致的问题。

5. 解决内存泄漏和堆栈溢出问题：在开发过程中，我们可能会遇到内存泄漏和堆栈溢出等问题，这些问题会导致程序的性能下降甚至崩溃。我们学会了如何使用工具和技术来检测和解决这些问题。通过使用内存检测工具和合理的内存管理策略，我们能够减少内存泄漏的发生。而对于堆栈溢出问题，我们学会了如何正确地分配和使用栈空间，以避免溢出导致的程序崩溃。

通过这次实验，我们不仅学会了如何使用Qt6和OpenCV进行图像识别软件的开发，还获得了优化算法、导入外部库、配置环境、打包可执行文件和多线程编程的经验。同时，我们也提升了对内存泄漏和堆栈溢出等常见问题的识别和解决能力，这些经验将对我们未来的软件开发工作非常有帮助。

