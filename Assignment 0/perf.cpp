#include <unistd.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
using namespace std;

int fibonacci(int n)
{
    if(n == 0 || n == 1)
    {
        return n;
    }
    else
    {
        return fibonacci(n-1)+fibonacci(n-2);
    }
}

int main()
{
    cout << "Running program" << endl;
    for(int val = 1; val <= 22;val++)
    {
        fibonacci(val);
    }
    cout << "Value computed " << endl;
    return 0;
}