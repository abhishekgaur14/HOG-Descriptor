#include "cuda.h"
#include <stdio.h>
#include "cuda_runtime_api.h"
#include <math.h>
#include <unistd.h>

#define PI 3.14159265
#define TILE_WIDTH 16

#define Mask_width 3
#define Mask_radius Mask_width/2
#define w (TILE_WIDTH + Mask_width - 1)


extern "C" {
#include "global.h"
}


__global__ void gradient_kernel(float* src_ptr, float* dst_ptr, float* theta_ptr, int rows, int cols)
{
    __shared__ float image[w][w];
    

    int width = cols;
    int height = rows;
    int tx = threadIdx.x;
    int ty = threadIdx.y;     //Thread ID's

    int colIdx   = blockDim.x * blockIdx.x + threadIdx.x;
    int rowIdx   = blockDim.y * blockIdx.y + threadIdx.y;

    int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int destY = dest / w;
    int destX = dest % w;
    int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
    int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
    int src = (srcY * width + srcX);
    
    if(srcY>=0 && srcY<height && srcX>=0 && srcX <width)
        image[destY][destX] = src_ptr[src];
    else
        image[destY][destX] = 0;

    dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH*TILE_WIDTH;
    destY = dest / w;
    destX = dest % w;
    srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
    srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
    src = srcY * width + srcX;

    if(destY < w)
    {
        if(srcY >= 0 && srcY < height && srcX >=0 && srcX < width)
            image[destY][destX] = src_ptr[src];
        else
            image[destY][destX] = 0;
    }
    
    __syncthreads();
    
    float x1,x2,theta_local;
    int x = tx + 1;
    int y = ty + 1;
    x1 = image[y][x+1] - image[y][x-1];
    x2 = image[y+1][x] - image[y-1][x];
    dst_ptr[rowIdx*cols + colIdx] = sqrt(x1*x1 + x2*x2);
    theta_local = atan2(x2,x1)*180/PI;
    if(theta_local < 0)
        theta_local = theta_local + 360;
    theta_ptr[rowIdx*cols + colIdx] = theta_local;
}


__global__ void d_compute_desc_kernel(float* mag_d, float* theta_d, float* blocks_desc_d, int rows, int cols)
{
   
    volatile __shared__ float s_block[32][4][NBINS];
  //  volatile __shared__ float   s_squares[4];
    
    const int cellIdx = threadIdx.x;    //cell in a block: 0-3
    const int columnIdx = threadIdx.y;  //which column for the particular thread 0-7
    const int sIdx = threadIdx.y*blockDim.x + threadIdx.x;     //The actual threadid out of 32

    // position of the upper-most pixel in the column for this thread
    const int blockX = (cellIdx % 2)*HOG_CELL_SIZE + columnIdx;
    const int blockY = cellIdx < 2 ? 0 : HOG_CELL_SIZE;

    const int pixelX = blockIdx.x * (HOG_BLOCK_WIDTH/2) + blockX;       // we assume 50% overlap
    const int pixelY = blockIdx.y * (HOG_BLOCK_HEIGHT/2) + blockY;

    // initialize all bins for this thread
    for(int i=0; i < NBINS; i++) 
    {
        for(int cell =0; cell < HOG_BLOCK_CELLS_X*HOG_BLOCK_CELLS_Y; cell++)
            s_block[sIdx][cell][i] = 0.f;
    }
    __syncthreads();

//<---------------------------------------------------------------------------------------------------------------------------------------->

    if(pixelX < cols && pixelY < rows)
    {
        for(int i=0; i<HOG_CELL_SIZE; i++)
        {
            const int pixelIdx = (pixelY + i)*cols + pixelX;


            float contribution = mag_d[pixelIdx];

            float binSize = 360.f/NBINS;

            float orientation = theta_d[pixelIdx] - binSize/2.f;;
            if(orientation < 0)
                orientation += 360.f;

            float delta = (orientation * NBINS)/360.f;

            int leftBin = (int)floorf(delta);
            delta -= leftBin;
            int rightBin = leftBin >= (NBINS-1) ? 0 : leftBin + 1;
            if(leftBin < 0)
                leftBin = NBINS-1;

            float rightContribution = contribution * delta;
            float leftContribution = contribution * (1-delta);

            s_block[sIdx][0][leftBin] += leftContribution;
            s_block[sIdx][0][rightBin]+= rightContribution;
            
            s_block[sIdx][1][leftBin] += leftContribution;
            s_block[sIdx][1][rightBin]+= rightContribution;

            s_block[sIdx][2][leftBin] += leftContribution;
            s_block[sIdx][2][rightBin]+= rightContribution;

            s_block[sIdx][3][leftBin] += leftContribution;
            s_block[sIdx][3][rightBin]+= rightContribution;

      }

    }

    __syncthreads();

//<------------------------------------------------------------------------------------------------------------------------>

    if(threadIdx.y == 0);
    {
        for(int i=1; i<32; i++)
        {
            for(int bin=0; bin<NBINS; bin++)
            {
                s_block[0][threadIdx.x][bin] += s_block[i][threadIdx.x][bin];
            }
        }
    }

    __syncthreads();
/*
// normalize the block histogram - L2+Hys normalization

    const float epsilon = 0.036f * 0.036f;  // magic numbers
    const float eHys    = 0.1f * 0.1f;
    const float clipThreshold = 0.2f;

    if(threadIdx.y == 0 ) 
    {
        float ls = 0.f;
        for(int j=0; j < NBINS; j++) 
        {
            ls += s_block[0][threadIdx.x][j] * s_block[0][threadIdx.x][j];
        }
        s_squares[threadIdx.x] = ls;
    }
    
    __syncthreads();
    if(threadIdx.y == 0 && threadIdx.x == 0 ) 
    {
        s_squares[0] += s_squares[1] + s_squares[2] + s_squares[3];
    }
    
    __syncthreads();
    // we use rsqrtf (reciprocal sqrtf) because of CUDA pecularities
    float normalization = rsqrtf(s_squares[0]+epsilon);
    // normalize and clip
    if(threadIdx.y == 0 ) 
    {
        for(int j=0; j < NBINS; j++)
        {
            s_block[0][threadIdx.x][j] *= normalization;
            s_block[0][threadIdx.x][j] = s_block[0][threadIdx.x][j] > clipThreshold ? clipThreshold : s_block[0][threadIdx.x][j];
        }
    }
    
    // renormalize
    if(threadIdx.y == 0 ) 
    {
        float ls = 0.f;
        for(int j=0; j < NBINS; j++) 
        {
            ls += s_block[0][threadIdx.x][j] * s_block[0][threadIdx.x][j];
        }
        s_squares[threadIdx.x] = ls;
    }
    __syncthreads();
    if(threadIdx.y == 0 && threadIdx.x == 0 ) 
    {
        s_squares[0] += s_squares[1] + s_squares[2] + s_squares[3];
    }

    normalization = rsqrtf(s_squares[0]+eHys);
    if(threadIdx.y == 0 ) 
    {
        for(int j=0; j < NBINS; j++) 
        {
            s_block[0][threadIdx.x][j] *= normalization;
        }
    }
*/





    if(threadIdx.y == 0 ) 
    {
        const int writeIdx = NBINS*4 * (blockIdx.y * gridDim.x + blockIdx.x);
        for(int bin=0; bin < NBINS; bin++) 
        {
            //printf("In saving part\n");
            blocks_desc_d[writeIdx + threadIdx.x*NBINS + bin] = s_block[0][threadIdx.x][bin];
            if(writeIdx + threadIdx.x*NBINS + bin == 0)
                printf("Value: %f\n",blocks_desc_d[writeIdx + threadIdx.x*NBINS + bin]);
        }
    }
}


extern "C" void gradient_kernel_caller(float* img_h, float* mag_h, float* hog_descriptor, int rows, int cols)
{
    float* img_d, *mag_d, *theta_d;

    cudaMalloc((void **) &img_d, rows*cols*sizeof(float));
    cudaMalloc((void **) &mag_d, rows*cols*sizeof(float));
    cudaMalloc((void **) &theta_d, rows*cols*sizeof(float));
    
    //int i;

    cudaMemcpy(img_d, img_h, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
    
    const int bl_size_x = 16;
    const int bl_size_y = 16;

    dim3 threads(bl_size_x, bl_size_y);
    dim3 grid((int)ceil(rows/(float)bl_size_x), (int)ceil(cols/(float)bl_size_y));
    
    gradient_kernel<<<grid,threads>>>(img_d, mag_d, theta_d, rows, cols);

    cudaMemcpy(mag_h, mag_d, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
//    cudaMemcpy(theta_h, theta_d, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);
    
    float* blocks_desc_d;

    const int nBlocks = ((rows/8)-1)*((cols/8)-1);
    const int total_blocks_size = nBlocks * HOG_BLOCK_CELLS_X * HOG_BLOCK_CELLS_Y * NBINS * sizeof(float);

    cudaMalloc((void**)&blocks_desc_d, total_blocks_size);

    dim3 dimGrid;

    dimGrid.x = (int)floor(cols/8.f)-1;
    dimGrid.y = (int)floor(rows/8.f)-1;


    dim3 dimBlock(4,8);
    printf("Grid: %d\t%d\n", dimBlock.x, dimBlock.y);

    d_compute_desc_kernel<<<dimGrid, dimBlock>>>(mag_d, theta_d, blocks_desc_d, rows, cols);
    cudaMemcpy(hog_descriptor, blocks_desc_d, total_blocks_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();   
    cudaPeekAtLastError();
    
    printf("After the second kernel\n");   
    
    /*int i,j;
    for(i=0;i<rows;i++)
    {
        for(j=0;j<cols;j++)
        {
            printf("%d\t%f\n", i*cols+j,theta_h[i*cols+j]);
        }
    }*/
}











