all: HOG_main


HOG_main: CUDA_kernel_module.o HOG_main.cpp
	g++ HOG_main.cpp CUDA_kernel_module.o -o HOG_main `pkg-config --cflags --libs opencv` -lcuda -lcudart -L/share/pkg/cuda/7.5.18/install/lib64/


CUDA_kernel_module.o: CUDA_kernel_module.cu
	nvcc CUDA_kernel_module.cu -c

clean:
	rm -rf *.o HOG_main
