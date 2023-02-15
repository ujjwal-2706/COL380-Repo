all: main.cpp library.hpp library.so
	g++ -c main.cpp -O3 -o main.o -fopenmp
	g++ -L. main.o -lrary -o exec -fopenmp
run : exec
	LD_LIBRARY_PATH =. ./exec
clean:
	rm exec main.o
