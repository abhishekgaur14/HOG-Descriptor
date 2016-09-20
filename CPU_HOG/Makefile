all: HOG_main_CPU

HOG_main_CPU:
	g++ hog_main.cpp -o HOG_main_CPU `pkg-config --cflags --libs opencv`

clean:
	rm -rf *.o HOG_main_CPU
