#include <stdio.h>


#define HOG_BLOCK_CELLS_X 2
#define HOG_BLOCK_CELLS_Y 2

#define HOG_CELL_SIZE 8
#define HOG_BLOCK_WIDTH     (8*HOG_BLOCK_CELLS_X)
#define HOG_BLOCK_HEIGHT    (8*HOG_BLOCK_CELLS_Y)

#define NBINS 9

extern "C" void gradient_kernel_caller(float* img_h, float* mag_h, float* hog_descriptor, int rows, int cols);
