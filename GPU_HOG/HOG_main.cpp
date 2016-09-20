#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>  
#include <iostream>
#include <stdio.h>
#include <vector>
#include "opencv2/gpu/gpu.hpp"
#include "global.h"
#include <iterator>
#include <time.h>

#define PI 3.14159265

#define GIG 1000000000 

using namespace std;
using namespace cv;


Mat get_hogdescriptor_visual_image(Mat& origImg,
                                   vector<float>& descriptorValues,
                                   Size winSize,
                                   Size cellSize,                                   
                                   int scaleFactor,
                                   double viz_factor);


int main()
{

    int check = cv::gpu::getCudaEnabledDeviceCount();

    cout<<"Check: "<< check<< endl;
    Mat img = imread("Car.jpg",0);
    if(!img.data)
        return -1;

	struct timespec diff(struct timespec start, struct timespec end);
	struct timespec time1, time2;
	struct timespec time_stamp;
    resize(img, img, Size(1024,1024));
    img.convertTo(img, CV_32FC1);
    img *= 1./255;
    pow(img, 0.5, img);

    Mat img2 = Mat::zeros(img.size(), CV_32FC1);

    Mat mat;
    mat.create(img.size(), CV_32FC1);

    img.copyTo(mat);
 
    cout<< "Step: "<< mat.step<< endl;   

    float* img_h = (float*)mat.data;
   
    Mat mag_final, theta_final;
    mag_final.create(img.size(), CV_32FC1);
    theta_final.create(img.size(), CV_32FC1);
    //float* mag_h = (float*)malloc(img.rows*img.cols*sizeof(float));
    float* mag_h = (float*)mag_final.data;
    float* theta_h = (float*)theta_final.data;

    const int nBlocks = ((img.rows/8)-1)*((img.cols/8)-1);
    int temp_size = nBlocks * HOG_BLOCK_CELLS_X * HOG_BLOCK_CELLS_Y * NBINS;
    const int total_blocks_size = nBlocks * HOG_BLOCK_CELLS_X * HOG_BLOCK_CELLS_Y * NBINS * sizeof(float);
    
    float* hog_check = (float*)malloc(total_blocks_size);
    
    
    clock_gettime(CLOCK_REALTIME, &time1);
    gradient_kernel_caller(img_h, mag_h, hog_check, img.rows, img.cols);

//    for(int i=0; i<temp_size; i++)
//    {
//       printf("Checking from host: %d\t%f\n", i, hog_check[i]);
//    }

    vector<float> hog_descriptor(hog_check, hog_check+temp_size);


    std::vector<float>::iterator result;
    result = max_element(hog_descriptor.begin(), hog_descriptor.end());
    int max_loc = distance(hog_descriptor.begin(), result);
    result = min_element(hog_descriptor.begin(), hog_descriptor.end());
    int min_loc = distance(hog_descriptor.begin(), result);

    float max_val = hog_descriptor[max_loc];
    float min_val = hog_descriptor[min_loc];
    vector<float> temp(hog_descriptor.size());
    for(int i=0; i<hog_descriptor.size(); i++)
    {
        temp[i] = (hog_descriptor[i] - min_val)/(max_val - min_val);
    }

    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp = diff(time1, time2);
    cout << "Time Taken: " << (float) ((GIG * time_stamp.tv_sec + time_stamp.tv_nsec)/1000000) <<"msec" << endl;
//    for(int i=0; i<temp.size(); i++)
//    {
//        printf("Checking from host: %d\t%f\n", i, temp[i]);
//    }

//    imshow("Checking", mag_final);
//    waitKey(0);
    //hog visualization  
    Mat r1 = get_hogdescriptor_visual_image(img2, 
                                    temp,  
                                    img2.size(),  
                                    Size(8,8),                                     
                                    1,  
                                    6);  
    
    transpose(r1,r1);
    imshow("hog visualization", r1);  
    //imwrite("HOG.jpg", r1);
    waitKey(0);  
       
    return 0;
}

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}




// HOGDescriptor visual_imagealizer
// adapted for arbitrary size of feature sets and training images
Mat get_hogdescriptor_visual_image(Mat& origImg,
                                   vector<float>& descriptorValues,
                                   Size winSize,
                                   Size cellSize,                                   
                                   int scaleFactor,
                                   double viz_factor)
{   
    Mat visual_image;
    resize(origImg, visual_image, Size(origImg.cols*scaleFactor, origImg.rows*scaleFactor));
 
    int gradientBinSize = 9;
    // dividing 180° into 9 bins, how large (in rad) is one bin?
    float radRangeForOneBin = 3.14/(float)gradientBinSize; 
 
    // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;
    int cells_in_y_dir = winSize.height / cellSize.height;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
 
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
 
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
 
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
 
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)            
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
 
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
 
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
 
                } // for (all bins)
 
 
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;
 
            } // for (all cells)
 
 
        } // for (all block x pos)
    } // for (all block y pos)
 
 
    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
 
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
 
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
 
 
    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
 
    // draw cells
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize.width;
            int drawY = celly * cellSize.height;
 
            int mx = drawX + cellSize.width/2;
            int my = drawY + cellSize.height/2;
 
            rectangle(visual_image,
                      Point(drawX*scaleFactor,drawY*scaleFactor),
                      Point((drawX+cellSize.width)*scaleFactor,
                      (drawY+cellSize.height)*scaleFactor),
                      CV_RGB(100,100,100),
                      1);
 
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
 
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
 
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
 
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = cellSize.width/2;
                float scale = viz_factor; // just a visual_imagealization scale,
                                          // to see the lines better
 
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
 
                // draw gradient visual_imagealization
                line(visual_image,
                     Point(x1*scaleFactor,y1*scaleFactor),
                     Point(x2*scaleFactor,y2*scaleFactor),
                     CV_RGB(0,0,255),
                     1);
 
            } // for (all bins)
 
        } // for (cellx)
    } // for (celly)
 
 
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
      for (int x=0; x<cells_in_x_dir; x++)
      {
           delete[] gradientStrengths[y][x];            
      }
      delete[] gradientStrengths[y];
      delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
 
    return visual_image;
 
}











