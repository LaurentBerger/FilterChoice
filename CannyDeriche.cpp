
#include<opencv2/opencv.hpp>
#include "opencv2/core/ocl.hpp"
#include<iostream>
using namespace cv;


using namespace cv;

/*
Using Canny's Criteria to Derive a Recursively Implemented Optimal Edge Detector International Journal of Computer Vision,167-187 (1987)
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.476.5736&rep=rep1&type=pdf
http://www.esiee.fr/~coupriem/Algo/algoima.html
*/



#include "opencv2/highgui.hpp"
#include <math.h>
#include <vector>
#include <iostream>


 





/*
Using Canny's Criteria to Derive a Recursively Implemented Optimal Edge Detector International Journal of Computer Vision,167-187 (1987)
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.476.5736&rep=rep1&type=pdf
*/

class ParallelGradientDericheYCols: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &im1;
    double alphaDerive;
    bool verbose;
public:
    ParallelGradientDericheYCols(Mat& imgSrc, Mat &d,double ald):
        img(imgSrc),
        im1(d),
        alphaDerive(ald),
        verbose(false)
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;

        float                *f2;
        int tailleSequence=(img.rows>img.cols)?img.rows:img.cols;
        double *g1=new double[tailleSequence],*g2=new double[tailleSequence];
        double    kp=pow(1-exp(-alphaDerive),2.0)/exp(-alphaDerive);
        double a1,a2,a3,a4;
        double b1,b2;
        int rows=img.rows,cols=img.cols;

        kp=pow(1-exp(-alphaDerive),2.0)/exp(-alphaDerive);
        a1=0;
        a2=kp*exp(-alphaDerive),a3=-kp*exp(-alphaDerive);
        a4=0;
        b1=2*exp(-alphaDerive);
        b2=-exp(-2*alphaDerive);

        switch(img.depth()){
        case CV_8U :
        {
            unsigned char *c1;
            for (int j=range.start;j<range.end;j++)
            {
                // Causal vertical  IIR filter
                c1 = (unsigned char*)img.ptr(0);
                f2 = (float*)im1.ptr(0);
                f2 += j;
                c1+=j;
                int i=0;
                g1[i] = (a1 + a2 +b1+b2)* *c1  ;
                g1[i] = (a1 + a2 )* *c1  ;
                i++;
                c1+=cols;
                g1[i] = a1 * *c1 + a2 * c1[-cols] + (b1+b2) * g1[i-1];
                g1[i] = a1 * *c1 + a2 * c1[-cols] + (b1) * g1[i-1];
                i++;
                c1+=cols;
                for (i=2;i<rows;i++,c1+=cols)
                    g1[i] = a1 * *c1 + a2 * c1[-cols] +b1*g1[i-1]+b2 *g1[i-2];
                // Anticausal vertical IIR filter
                c1 = (unsigned char*)img.ptr(0);
                c1 += (rows-1)*cols+j;
                i = rows-1;
                g2[i] =(a3+a4+b1+b2)* *c1;
                g2[i] =(a3+a4)* *c1;
                i--;
                c1-=cols;
                g2[i] = a3* c1[cols] + a4 * c1[cols]+(b1+b2)*g2[i+1];
                g2[i] = a3* c1[cols] + a4 * c1[cols]+(b1)*g2[i+1];
                i--;
                c1-=cols;
                for (i=rows-3;i>=0;i--,c1-=cols)
                    g2[i] = a3*c1[cols] +a4* c1[2*cols]+
                            b1*g2[i+1]+b2*g2[i+2];
                for (i=0;i<rows;i++,f2+=cols)
                    *f2 = (float)(g1[i]+g2[i]);
                }
            }
            break;
        case CV_16S :
        case CV_16U :
        {
            unsigned short *c1;
            for (int j=range.start;j<range.end;j++)
            {
                c1 = ((unsigned short*)img.ptr(0));
                f2 = ((float*)im1.ptr(0));
                f2 += j;
                c1+=j;
                int i=0;
                g1[i] = (a1 + a2 +b1+b2)* *c1  ;
                g1[i] = (a1 + a2 )* *c1  ;
                i++;
                c1+=cols;
                g1[i] = a1 * *c1 + a2 * c1[-cols] + (b1+b2) * g1[i-1];
                g1[i] = a1 * *c1 + a2 * c1[-cols] + (b1) * g1[i-1];
                i++;
                c1+=cols;
                for (i=2;i<rows;i++,c1+=cols)
                    g1[i] = a1 * *c1 + a2 * c1[-cols] +b1*g1[i-1]+b2 *g1[i-2];
                // Anticausal vertical IIR filter
                c1 = ((unsigned short*)img.ptr(0));
                c1 += (rows-1)*cols+j;
                i = rows-1;
                g2[i] =(a3+a4+b1+b2)* *c1;
                g2[i] =(a3+a4)* *c1;
                i--;
                c1-=cols;
                g2[i] = (a3+a4)* c1[cols] +(b1+b2)*g2[i+1];
                g2[i] = (a3+a4)* c1[cols] +(b1)*g2[i+1];
                i--;
                c1-=cols;
                for (i=rows-3;i>=0;i--,c1-=cols)
                    g2[i] = a3*c1[cols] +a4* c1[2*cols]+b1*g2[i+1]+b2*g2[i+2];
                c1 = ((unsigned short*)img.ptr(0))+j;
                for (i=0;i<rows;i++,f2+=cols,c1+=cols)
                    *f2 = 0**c1+(float)(g1[i]+g2[i]);
            }
        }
            break;
        case CV_32S :
             break;
        case CV_32F :
             break;
        case CV_64F :
             break;
        default :
            delete []g1;
            delete []g2;
            return ;
            }
        delete []g1;
        delete []g2;
    };
    ParallelGradientDericheYCols& operator=(const ParallelGradientDericheYCols &) {
         return *this;
    };
};


class ParallelGradientDericheYRows: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double alphaMoyenne;
    bool verbose;

public:
    ParallelGradientDericheYRows(Mat& imgSrc, Mat &d,double alm):
        img(imgSrc),
        dst(d),
        alphaMoyenne(alm),
        verbose(false)
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;
        float *f1,*f2;
        int tailleSequence=(img.rows>img.cols)?img.rows:img.cols;
        double *g1=new double[tailleSequence],*g2=new double[tailleSequence];
        double k,a5,a6,a7,a8;
        double b3,b4;
        int cols=img.cols;

        k=pow(1-exp(-alphaMoyenne),2.0)/(1+2*alphaMoyenne*exp(-alphaMoyenne)-exp(-2*alphaMoyenne));
        a5=k;
        a6=k*exp(-alphaMoyenne)*(alphaMoyenne-1);
        a7=k*exp(-alphaMoyenne)*(alphaMoyenne+1);
        a8=-k*exp(-2*alphaMoyenne);
        b3=2*exp(-alphaMoyenne);
        b4=-exp(-2*alphaMoyenne);

        for (int i=range.start;i<range.end;i++)
            {
            f2 = ((float*)dst.ptr(i));
            f1 = ((float*)img.ptr(i));
            int j=0;
            g1[j] = (a5 +a6+b3+b4)* *f1 ;
            g1[j] = (a5 +a6)* *f1 ;
            j++;
            f1++;
            g1[j] = a5 * f1[0]+a6*f1[j-1]+(b3+b4) * g1[j-1];
            g1[j] = a5 * f1[0]+a6*f1[j-1]+(b3) * g1[j-1];
            j++;
            f1++;
            for (j=2;j<cols;j++,f1++)
                g1[j] = a5 * f1[0] + a6 * f1[-1]+b3*g1[j-1]+b4*g1[j-2];
            f1 = ((float*)img.ptr(0));
            f1 += i*cols+cols-1;
            j=cols-1;
            g2[j] = (a7+a8+b3+b4)* *f1;
            g2[j] = (a7+a8)* *f1;
            j--;
            f1--;
            g2[j] = (a7+a8) * f1[1]  +(b3+b4) * g2[j+1];
            g2[j] = (a7+a8) * f1[1]  +(b3) * g2[j+1];
            j--;
            f1--;
            for (j=cols-3;j>=0;j--,f1--)
                g2[j] = a7*f1[1]+a8*f1[2]+b3*g2[j+1]+b4*g2[j+2];
            for (j=0;j<cols;j++,f2++)
                *f2 = (float)(g1[j]+g2[j]);
            }
        delete []g1;
        delete []g2;

    };
    ParallelGradientDericheYRows& operator=(const ParallelGradientDericheYRows &) {
         return *this;
    };
};


class ParallelGradientDericheXCols: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double alphaMoyenne;
    bool verbose;

public:
    ParallelGradientDericheXCols(Mat& imgSrc, Mat &d,double alm):
        img(imgSrc),
        dst(d),
        alphaMoyenne(alm),
        verbose(false)
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {

        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;
        float                *f1,*f2;
        int rows=img.rows,cols=img.cols;

        int tailleSequence=(rows>cols)?rows:cols;
        double *g1=new double[tailleSequence],*g2=new double[tailleSequence];
        double k,a5,a6,a7,a8=0;
        double b3,b4;

        k=pow(1-exp(-alphaMoyenne),2.0)/(1+2*alphaMoyenne*exp(-alphaMoyenne)-exp(-2*alphaMoyenne));
        a5=k,a6=k*exp(-alphaMoyenne)*(alphaMoyenne-1);
        a7=k*exp(-alphaMoyenne)*(alphaMoyenne+1),a8=-k*exp(-2*alphaMoyenne);
        b3=2*exp(-alphaMoyenne);
        b4=-exp(-2*alphaMoyenne);

        for (int j=range.start;j<range.end;j++)
        {
            f1 = (float*)img.ptr(0);
            f1+=j;
            int i=0;
            g1[i] = (a5 + a6 +b3+b4)* *f1  ;
            g1[i] = (a5 + a6 )* *f1  ;
            i++;
            f1+=cols;
            g1[i] = a5 * *f1 + a6 * f1[-cols] + (b3+b4) * g1[i-1];
            g1[i] = a5 * *f1 + a6 * f1[-cols] + (b3) * g1[i-1];
            i++;
            f1+=cols;
            for (i=2;i<rows;i++,f1+=cols)
                g1[i] = a5 * *f1 + a6 * f1[-cols] +b3*g1[i-1]+b4 *g1[i-2];
            f1 = (float*)img.ptr(0);
            f1 += (rows-1)*cols+j;
            i = rows-1;
            g2[i] =(a7+a8+b3+b4)* *f1;
            g2[i] =(a7+a8)* *f1;
            i--;
            f1-=cols;
            g2[i] = (a7+a8)* f1[cols] +(b3+b4)*g2[i+1];
            g2[i] = (a7+a8)* f1[cols] +(b3)*g2[i+1];
            i--;
            f1-=cols;
            for (i=rows-3;i>=0;i--,f1-=cols)
                g2[i] = a7*f1[cols] +a8* f1[2*cols]+
                        b3*g2[i+1]+b4*g2[i+2];
            for (i=0;i<rows;i++,f2+=cols)
            {
                f2 = ((float*)dst.ptr(i))+(j*img.channels());
                *f2 = (float)(g1[i]+g2[i]);
            }
        }
        delete []g1;
        delete []g2;
    };
    ParallelGradientDericheXCols& operator=(const ParallelGradientDericheXCols &) {
         return *this;
    };
};


class ParallelGradientDericheXRows: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double alphaDerive;
    bool verbose;

public:
    ParallelGradientDericheXRows(Mat& imgSrc, Mat &d,double ald):
        img(imgSrc),
        dst(d),
        alphaDerive(ald),
        verbose(false)
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;
        float *f1;
        int rows=img.rows,cols=img.cols;
        int tailleSequence=(rows>cols)?rows:cols;
        double *g1=new double[tailleSequence],*g2=new double[tailleSequence];
        double kp;;
        double a1,a2,a3,a4;
        double b1,b2;

        kp=pow(1-exp(-alphaDerive),2.0)/exp(-alphaDerive);
        a1=0;
        a2=kp*exp(-alphaDerive);
        a3=-kp*exp(-alphaDerive);
        a4=0;
        b1=2*exp(-alphaDerive);
        b2=-exp(-2*alphaDerive);

        switch(img.depth()){
        case CV_8U :
        case CV_8S :
            {
            unsigned char *c1;
            for (int i=range.start;i<range.end;i++)
                {
                f1 = (float*)dst.ptr(i);
                c1 = (unsigned char*)img.ptr(i);
                int j=0;
                g1[j] = (a1 +a2+b1+b2)* *c1 ;
                g1[j] = (a1 +a2)* *c1 ;
                j++;
                c1++;
                g1[j] = a1 * c1[0]+a2*c1[j-1]+(b1+b2) * g1[j-1];
                g1[j] = a1 * c1[0]+a2*c1[j-1]+(b1) * g1[j-1];
                j++;
                c1++;
                for (j=2;j<cols;j++,c1++)
                    g1[j] = a1 * c1[0] + a2 * c1[-1]+b1*g1[j-1]+b2*g1[j-2];
                c1 = (unsigned char*)img.ptr(0);
                c1 += i*cols+cols-1;
                j=cols-1;
                g2[j] = (a3+a4+b1+b2)* *c1;
                g2[j] = (a3+a4)* *c1;
                j--;
                g2[j] = (a3+a4) * c1[1]  +(b1+b2) * g2[j+1];
                g2[j] = (a3+a4) * c1[1]  +(b1) * g2[j+1];
                j--;
                c1--;
                for (j=cols-3;j>=0;j--,c1--)
                    g2[j] = a3*c1[1]+a4*c1[2]+b1*g2[j+1]+b2*g2[j+2];
                for (j=0;j<cols;j++,f1++)
                    *f1 = (float)(g1[j]+g2[j]);
                }
            }
            break;
        case CV_16S :
        case CV_16U :
            {
            unsigned short *c1;
            f1 = ((float*)dst.ptr(0));
            for (int i=range.start;i<range.end;i++)
                {
                c1 = ((unsigned short*)img.ptr(0));
                c1 += i*cols;
                int j=0;
                g1[j] = (a1 +a2+b1+b2)* *c1 ;
                g1[j] = (a1 +a2)* *c1 ;
                j++;
                c1++;
                g1[j] = a1 * c1[0]+a2*c1[j-1]+(b1+b2) * g1[j-1];
                g1[j] = a1 * c1[0]+a2*c1[j-1]+(b1) * g1[j-1];
                j++;
                c1++;
                for (j=2;j<cols;j++,c1++)
                    g1[j] = a1 * c1[0] + a2 * c1[-1]+b1*g1[j-1]+b2*g1[j-2];
                c1 = ((unsigned short*)img.ptr(0));
                c1 += i*cols+cols-1;
                j=cols-1;
                g2[j] = (a3+a4+b1+b2)* *c1;
                g2[j] = (a3+a4)* *c1;
                j--;
                c1--;
                g2[j] = (a3+a4) * c1[1]  +(b1+b2) * g2[j+1];
                g2[j] = (a3+a4) * c1[1]  +(b1) * g2[j+1];
                j--;
                c1--;
                for (j=cols-3;j>=0;j--,c1--)
                    g2[j] = a3*c1[1]+a4*c1[2]+b1*g2[j+1]+b2*g2[j+2];
                for (j=0;j<cols;j++,f1++)
                    *f1 = (float)(g1[j]+g2[j]);
                }
            }
            break;
        default :
            return ;
            }
        delete []g1;
        delete []g2;
    };
    ParallelGradientDericheXRows& operator=(const ParallelGradientDericheXRows &) {
         return *this;
    };
};

UMat GradientDericheY(UMat op, double alphaDerive,double alphaMean)
{
    Mat tmp(op.size(),CV_32FC(op.channels()));
    UMat imDst(op.rows,op.cols,CV_32FC(op.channels()));
    cv::Mat opSrc = op.getMat(cv::ACCESS_RW);
    cv::Mat dst = imDst.getMat(cv::ACCESS_RW);
    std::vector<Mat> planSrc;
    split(opSrc,planSrc);
    std::vector<Mat> planTmp;
    split(tmp,planTmp);
    std::vector<Mat> planDst;
    split(dst,planDst);
    for (int i = 0; i < static_cast<int>(planSrc.size()); i++)
    {
        if (planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous())
        {
            ParallelGradientDericheYCols x(planSrc[i],planTmp[i],alphaDerive);
            parallel_for_(Range(0,opSrc.cols), x,getNumThreads());
            ParallelGradientDericheYRows xr(planTmp[i],planDst[i],alphaMean);
            parallel_for_(Range(0,opSrc.rows), xr,getNumThreads());

        }
        else
            std::cout << "PB";
    }
    merge(planDst,imDst);
    return imDst;
}

UMat GradientDericheX(UMat op, double alphaDerive,double alphaMean)
{
    Mat tmp(op.size(),CV_32FC(op.channels()));
    UMat imDst(op.rows,op.cols,CV_32FC(op.channels()));
    cv::Mat opSrc = op.getMat(cv::ACCESS_RW);
    cv::Mat dst = imDst.getMat(cv::ACCESS_RW);
    std::vector<Mat> planSrc;
    split(opSrc,planSrc);
    std::vector<Mat> planTmp;
    split(tmp,planTmp);
    std::vector<Mat> planDst;
    split(dst,planDst);
    for (int i = 0; i < static_cast<int>(planSrc.size()); i++)
    {
        if (planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous())
        {
            ParallelGradientDericheXRows x(planSrc[i],planTmp[i],alphaDerive);
            parallel_for_(Range(0,opSrc.rows), x,getNumThreads());
            ParallelGradientDericheXCols xr(planTmp[i],planDst[i],alphaMean);
            parallel_for_(Range(0,opSrc.cols), xr,getNumThreads());
        }
        else
            std::cout << "PB";
    }
    merge(planDst,imDst);
    return imDst;
}





void CannyBis(  OutputArray _dst,
                double low_thresh, double high_thresh,
                bool L2gradient ,InputOutputArray _dx,InputOutputArray _dy)
{
    const int type = _dx.type(), depth = CV_MAT_DEPTH(type), cn = 1;
    const Size size = _dx.size();

    CV_Assert( depth == CV_16S );
    _dst.create(size, CV_8U);

    if (!L2gradient && ( CV_CANNY_L2_GRADIENT) == CV_CANNY_L2_GRADIENT)
    {
        // backward compatibility
        L2gradient = true;
    }


    if (low_thresh > high_thresh)
        std::swap(low_thresh, high_thresh);


    Mat dst = _dst.getMat();


#ifdef HAVE_TBB

if (L2gradient)
{
    low_thresh = std::min(32767.0, low_thresh);
    high_thresh = std::min(32767.0, high_thresh);

    if (low_thresh > 0) low_thresh *= low_thresh;
    if (high_thresh > 0) high_thresh *= high_thresh;
}
int low = cvFloor(low_thresh);
int high = cvFloor(high_thresh);

ptrdiff_t mapstep = src.cols + 2;
AutoBuffer<uchar> buffer((src.cols+2)*(src.rows+2));

uchar* map = (uchar*)buffer;
memset(map, 1, mapstep);
memset(map + mapstep*(src.rows + 1), 1, mapstep);

int threadsNumber = tbb::task_scheduler_init::default_num_threads();
int grainSize = src.rows / threadsNumber;

// Make a fallback for pictures with too few rows.
uchar ksize2 = aperture_size / 2;
int minGrainSize = 1 + ksize2;
int maxGrainSize = src.rows - 2 - 2*ksize2;
if ( !( minGrainSize <= grainSize && grainSize <= maxGrainSize ) )
{
    threadsNumber = 1;
    grainSize = src.rows;
}

tbb::task_group g;

for (int i = 0; i < threadsNumber; ++i)
{
    if (i < threadsNumber - 1)
        g.run(tbbCanny(Range(i * grainSize, (i + 1) * grainSize), src, map, low, high, aperture_size, L2gradient));
    else
        g.run(tbbCanny(Range(i * grainSize, src.rows), src, map, low, high, aperture_size, L2gradient));
}

g.wait();

#define CANNY_PUSH_SERIAL(d)    *(d) = uchar(2), borderPeaks.push(d)

// now track the edges (hysteresis thresholding)
uchar* m;
while (borderPeaks.try_pop(m))
{
    if (!m[-1])         CANNY_PUSH_SERIAL(m - 1);
    if (!m[1])          CANNY_PUSH_SERIAL(m + 1);
    if (!m[-mapstep-1]) CANNY_PUSH_SERIAL(m - mapstep - 1);
    if (!m[-mapstep])   CANNY_PUSH_SERIAL(m - mapstep);
    if (!m[-mapstep+1]) CANNY_PUSH_SERIAL(m - mapstep + 1);
    if (!m[mapstep-1])  CANNY_PUSH_SERIAL(m + mapstep - 1);
    if (!m[mapstep])    CANNY_PUSH_SERIAL(m + mapstep);
    if (!m[mapstep+1])  CANNY_PUSH_SERIAL(m + mapstep + 1);
}

#else
    Mat dx,dy;
    dx = _dx.getMat(), dy = _dy.getMat();



    if (L2gradient)
    {
        low_thresh = std::min(32767.0, low_thresh);
        high_thresh = std::min(32767.0, high_thresh);

        if (low_thresh > 0) low_thresh *= low_thresh;
        if (high_thresh > 0) high_thresh *= high_thresh;
    }
    int low = cvFloor(low_thresh);
    int high = cvFloor(high_thresh);

    ptrdiff_t mapstep = dx.cols + 2;
    AutoBuffer<uchar> buffer((dx.cols+2)*(dx.rows+2) + cn * mapstep * 3 * sizeof(int));

    int* mag_buf[3];
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep*cn;
    mag_buf[2] = mag_buf[1] + mapstep*cn;
    memset(mag_buf[0], 0, /* cn* */mapstep*sizeof(int));

    uchar* map = (uchar*)(mag_buf[2] + mapstep*cn);
    memset(map, 1, mapstep);
    memset(map + mapstep*(dx.rows + 1), 1, mapstep);

    int maxsize = std::max(1 << 10, dx.cols * dx.rows / 10);
    std::vector<uchar*> stack(maxsize);
    uchar **stack_top = &stack[0];
    uchar **stack_bottom = &stack[0];

    /* sector numbers
       (Top-Left Origin)

        1   2   3
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        3   2   1
    */

    #define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
    #define CANNY_POP(d)     (d) = *--stack_top

#if CV_SSE2
    bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#endif

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = 0; i <= dx.rows; i++)
    {
        int* _norm = mag_buf[(i > 0) + 1] + 1;
        if (i < dx.rows)
        {
            short* _dx = dx.ptr<short>(i);
            short* _dy = dy.ptr<short>(i);

            if (!L2gradient)
            {
                int j = 0, width = dx.cols * cn;
#if CV_SSE2
                if (haveSSE2)
                {
                    __m128i v_zero = _mm_setzero_si128();
                    for ( ; j <= width - 8; j += 8)
                    {
                        __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                        __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));
                        v_dx = _mm_max_epi16(v_dx, _mm_sub_epi16(v_zero, v_dx));
                        v_dy = _mm_max_epi16(v_dy, _mm_sub_epi16(v_zero, v_dy));

                        __m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx, v_zero), _mm_unpacklo_epi16(v_dy, v_zero));
                        _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                        v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx, v_zero), _mm_unpackhi_epi16(v_dy, v_zero));
                        _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                    }
                }
#elif CV_NEON
                for ( ; j <= width - 8; j += 8)
                {
                    int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
                    vst1q_s32(_norm + j, vaddq_s32(vabsq_s32(vmovl_s16(vget_low_s16(v_dx))),
                                                   vabsq_s32(vmovl_s16(vget_low_s16(v_dy)))));
                    vst1q_s32(_norm + j + 4, vaddq_s32(vabsq_s32(vmovl_s16(vget_high_s16(v_dx))),
                                                       vabsq_s32(vmovl_s16(vget_high_s16(v_dy)))));
                }
#endif
                for ( ; j < width; ++j)
                    _norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
            }
            else
            {
                int j = 0, width = dx.cols * cn;
#if CV_SSE2
                if (haveSSE2)
                {
                    for ( ; j <= width - 8; j += 8)
                    {
                        __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                        __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));

                        __m128i v_dx_ml = _mm_mullo_epi16(v_dx, v_dx), v_dx_mh = _mm_mulhi_epi16(v_dx, v_dx);
                        __m128i v_dy_ml = _mm_mullo_epi16(v_dy, v_dy), v_dy_mh = _mm_mulhi_epi16(v_dy, v_dy);

                        __m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx_ml, v_dx_mh), _mm_unpacklo_epi16(v_dy_ml, v_dy_mh));
                        _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                        v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx_ml, v_dx_mh), _mm_unpackhi_epi16(v_dy_ml, v_dy_mh));
                        _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                    }
                }
#elif CV_NEON
                for ( ; j <= width - 8; j += 8)
                {
                    int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
                    int16x4_t v_dxp = vget_low_s16(v_dx), v_dyp = vget_low_s16(v_dy);
                    int32x4_t v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                    vst1q_s32(_norm + j, v_dst);

                    v_dxp = vget_high_s16(v_dx), v_dyp = vget_high_s16(v_dy);
                    v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                    vst1q_s32(_norm + j + 4, v_dst);
                }
#endif
                for ( ; j < width; ++j)
                    _norm[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];
            }

            if (cn > 1)
            {
                for(int j = 0, jn = 0; j < dx.cols; ++j, jn += cn)
                {
                    int maxIdx = jn;
                    for(int k = 1; k < cn; ++k)
                        if(_norm[jn + k] > _norm[maxIdx]) maxIdx = jn + k;
                    _norm[j] = _norm[maxIdx];
                    _dx[j] = _dx[maxIdx];
                    _dy[j] = _dy[maxIdx];
                }
            }
            _norm[-1] = _norm[dx.cols] = 0;
        }
        else
            memset(_norm-1, 0, /* cn* */mapstep*sizeof(int));

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i == 0)
            continue;

        uchar* _map = map + mapstep*i + 1;
        _map[-1] = _map[dx.cols] = 1;

        int* _mag = mag_buf[1] + 1; // take the central row
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        const short* _x = dx.ptr<short>(i-1);
        const short* _y = dy.ptr<short>(i-1);

        if ((stack_top - stack_bottom) + dx.cols > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = std::max(maxsize * 3/2, sz + dx.cols);
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        int prev_flag = 0;
        for (int j = 0; j < dx.cols; j++)
        {
            #define CANNY_SHIFT 15
            const int TG22 = (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);

            int m = _mag[j];

            if (m > low)
            {
                int xs = _x[j];
                int ys = _y[j];
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;

                int tg22x = x * TG22;

                if (y < tg22x)
                {
                    if (m > _mag[j-1] && m >= _mag[j+1]) goto __ocv_canny_push;
                }
                else
                {
                    int tg67x = tg22x + (x << (CANNY_SHIFT+1));
                    if (y > tg67x)
                    {
                        if (m > _mag[j+magstep2] && m >= _mag[j+magstep1]) goto __ocv_canny_push;
                    }
                    else
                    {
                        int s = (xs ^ ys) < 0 ? -1 : 1;
                        if (m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s]) goto __ocv_canny_push;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = uchar(1);
            continue;
__ocv_canny_push:
            if (!prev_flag && m > high && _map[j-mapstep] != 2)
            {
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while (stack_top > stack_bottom)
    {
        uchar* m;
        if ((stack_top - stack_bottom) + 8 > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3/2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if (!m[-1])         CANNY_PUSH(m - 1);
        if (!m[1])          CANNY_PUSH(m + 1);
        if (!m[-mapstep-1]) CANNY_PUSH(m - mapstep - 1);
        if (!m[-mapstep])   CANNY_PUSH(m - mapstep);
        if (!m[-mapstep+1]) CANNY_PUSH(m - mapstep + 1);
        if (!m[mapstep-1])  CANNY_PUSH(m + mapstep - 1);
        if (!m[mapstep])    CANNY_PUSH(m + mapstep);
        if (!m[mapstep+1])  CANNY_PUSH(m + mapstep + 1);
    }

#endif

    // the final pass, form the final image
    const uchar* pmap = map + mapstep + 1;
    uchar* pdst = dst.ptr();
    for (int i = 0; i < dx.rows; i++, pmap += mapstep, pdst += dst.step)
    {
        for (int j = 0; j < dx.cols; j++)
            pdst[j] = (uchar)-(pmap[j] >> 1);
    }
}




