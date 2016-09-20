#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>  
#include <iostream>
#include <stdio.h>
#include <vector>
#include <omp.h>
#include <time.h>

#define PI 3.14159265
#define GIG 1000000000

using namespace std;
using namespace cv;


void find_gradients(Mat img, Mat& mag, Mat& theta);
void image_feature(Mat mag, Mat theta, vector<float>& hog_descriptor);
void block_feature(Mat block_mag, Mat block_theta, vector<float>& hog_block_desc);
void cell_feature(Mat cell_mag, Mat cell_theta, vector<float>& hog_cell_desc);
Mat get_hogdescriptor_visual_image(Mat& origImg,
                                   vector<float>& descriptorValues,
                                   Size winSize,
                                   Size cellSize,                                   
                                   int scaleFactor,
                                   double viz_factor);

int main()
{
	Mat img = imread("Car.jpg",0);
	if(!img.data)
		return -1;
	
	struct timespec diff(struct timespec start, struct timespec end);
	struct timespec time1, time2;
	struct timespec time_stamp;
	resize(img, img, Size(1024,1024));
	Mat img1;
	img.copyTo(img1);
	Mat img2 = Mat::zeros(img.size(), CV_32FC1);
	img.convertTo(img, CV_32FC1);
	img *= 1./255;
	//Color Normalization
	pow(img, 0.5, img);
	Mat mag, theta;
	mag.create(img.size(), img.type());
	int i;	

	clock_gettime(CLOCK_REALTIME, &time1);
	find_gradients(img, mag, theta);
	//waitKey(0);

	vector<float> hog_descriptor;
	// cout<< "Before: "<< hog_descriptor.size()<< endl;
	image_feature(mag, theta, hog_descriptor);

	//Normalization
	std::vector<float>::iterator result;
	result = max_element(hog_descriptor.begin(), hog_descriptor.end());
	int max_loc = distance(hog_descriptor.begin(), result);
	result = min_element(hog_descriptor.begin(), hog_descriptor.end());
	int min_loc = distance(hog_descriptor.begin(), result);

	float max_val = hog_descriptor[max_loc];
	float min_val = hog_descriptor[min_loc];
	vector<float> temp(hog_descriptor.size());
	
	for(i=0; i<hog_descriptor.size(); i++)
	{
		temp[i] = (hog_descriptor[i] - min_val)/(max_val - min_val);
	}
	cout<< "Checking final size: "<< hog_descriptor.size()<< endl;
	clock_gettime(CLOCK_REALTIME, &time2);
	time_stamp = diff(time1, time2);
	
	//Checking with openCV HOG computation
	HOGDescriptor d(img.size(), Size(16,16), Size(8,8), Size(8,8), 9);
	vector<float> descriptorsValues;
	vector<Point> locations;
	d.compute( img1, descriptorsValues, Size(0,0), Size(0,0), locations);


	//hog visualization  
	Mat r1 = get_hogdescriptor_visual_image(img2, 
	                               temp,  
	                               img2.size(),  
	                               Size(8,8),                                     
	                               1,  
	                               6);  

	imshow("hog visualization", r1);  
	cout << "Time Taken: " << (float) ((GIG * time_stamp.tv_sec + time_stamp.tv_nsec)/1000000) <<"msec" << endl;
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



void find_gradients(Mat img, Mat& mag, Mat& theta)
{
	Mat hmask, vmask;
	hmask = (Mat_<float>(1,3) << -1, 0 ,1);
	vmask = (Mat_<float>(3,1) << -1 ,0 ,1);

	Mat grad_x, grad_y;
	grad_x.create(img.size(), CV_32FC1);
	grad_y.create(img.size(), CV_32FC1);
	for(int i=1; i<img.rows-1; i++)
	{
		for(int j=1; j<img.cols-1; j++)
		{
			grad_x.at<float>(i,j) = img.at<float>(i,j+1) - img.at<float>(i,j-1);
			grad_y.at<float>(i,j) = img.at<float>(i+1,j) - img.at<float>(i-1,j);
			float x1 = grad_x.at<float>(i,j);
			float x2 = grad_y.at<float>(i,j);
			mag.at<float>(i,j) = sqrt(x1*x1 + x2*x2);
		}
	}

	//filter2D(img, grad_x, img.depth(), hmask);

	//filter2D(img, grad_y, img.depth(), vmask);

	//cartToPolar(grad_x, grad_y, mag, theta);

//	magnitude(grad_x, grad_y, mag);
	phase(grad_x, grad_y, theta);
	theta = theta * 180 / PI; 
}

void image_feature(Mat mag, Mat theta, vector<float>& hog_descriptor)
{
	int Block_Size[2] = {16,16};
	int rowStart, rowEnd, colStart, colEnd;
	Mat block_mag, block_theta;
	int i,j;
	
//	#pragma omp parallel for private(i,j)
	for(i=0;i<(2*mag.cols/Block_Size[0])-1;i++)
	{
		vector<float> hog_block_desc;

		for(j=0; j<(2*mag.rows/Block_Size[1])-1;j++)
		{
			rowStart = i*(Block_Size[0]/2);
			rowEnd = (i+2)*(Block_Size[0]/2);
			colStart = j*(Block_Size[1]/2);
			colEnd = (j+2)*(Block_Size[1]/2);
			block_mag  = mag(Range(colStart, colEnd), Range(rowStart,rowEnd));
			block_theta  = theta(Range(colStart, colEnd), Range(rowStart,rowEnd));
			block_feature(block_mag, block_theta, hog_block_desc);
			hog_descriptor.insert(hog_descriptor.end(), hog_block_desc.begin(), hog_block_desc.end());
			hog_block_desc.clear();
		}
	}
}


void block_feature(Mat block_mag, Mat block_theta, vector<float>& hog_block_desc)
{
	int grid_size[2] = {2,2};

	int rowStart, rowEnd, colStart, colEnd;

	int cell_rows = block_mag.rows/grid_size[0];
	int cell_cols = block_mag.cols/grid_size[1];
	vector<float> hog_cell_desc(9);
	int i,j;

	Mat cell_mag, cell_theta;

//	#pragma omp parallel for private(i,j)
	for(i=0;i<grid_size[0]; i++)
	{
		for(j=0; j<grid_size[1]; j++)
		{
			rowStart = i*cell_rows;
			rowEnd = (i+1)*cell_rows;
			colStart = j*cell_cols;
			colEnd = (j+1)*cell_cols;
			cell_mag  = block_mag(Range(colStart, colEnd), Range(rowStart,rowEnd));
			cell_theta  = block_theta(Range(colStart, colEnd), Range(rowStart,rowEnd));
			cell_feature(cell_mag, cell_theta, hog_cell_desc);
			hog_block_desc.insert(hog_block_desc.end(), hog_cell_desc.begin(), hog_cell_desc.end());
			// cout<< "Checking"<< hog_block_desc.size()<< endl;
		}
	}
}


void cell_feature(Mat cell_mag, Mat cell_theta, vector<float>& hog_cell_desc)
{
	int Bins[9] = {20,60,100,140,180,220,260,300,340};

	GaussianBlur(cell_mag, cell_mag, Size(7,7), 0, 0);
	int i,j,k,l;
	// Converting 2d to 1d data
	vector<float> mag_vec, theta_vec;
	if(cell_mag.isContinuous())
		mag_vec.assign((float*)cell_mag.datastart, (float*)cell_mag.dataend);
	else
	{
		for(i=0; i<cell_mag.rows; i++)
		{
			mag_vec.insert(mag_vec.end(), (float*)cell_mag.ptr<uchar>(i), (float*)cell_mag.ptr<uchar>(i)+cell_mag.cols);
		}
	}

	if(cell_theta.isContinuous())
		theta_vec.assign((float*)cell_theta.datastart, (float*)cell_theta.dataend);
	else
	{
		for(j=0; j<cell_theta.rows; j++)
		{
			theta_vec.insert(theta_vec.end(), (float*)cell_theta.ptr<uchar>(j), (float*)cell_theta.ptr<uchar>(j)+cell_theta.cols);
		}
	}

	// hog_cell_desc.reserve(9);

	int count = 0;
	float angle, magnitude;
//	omp_set_num_threads();
//	#pragma omp parallel for private(k,l) 
	for(k=0; k< theta_vec.size(); k++)
	{
		angle = theta_vec[k];
		magnitude = mag_vec[k];

		if(angle > 340)
			hog_cell_desc[8] += magnitude*(angle-340)/40;
 		else if(angle <= 20)
 			hog_cell_desc[0] += magnitude*(20-angle)/40;
 		else
 		{
			for(l=0; l<8; l++)
			{
				if(angle > Bins[l] && angle <= Bins[l+1])
				{
					hog_cell_desc[l] += magnitude*(Bins[l+1]-angle)/40;
					hog_cell_desc[l+1] += magnitude*(angle-Bins[l])/40;
					break;
 				}
			}
 		}
	}
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
