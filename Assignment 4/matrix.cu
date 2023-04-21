#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <bits/stdc++.h>
using namespace std;

struct Block4
{
    int i,j;
    int blockValue[4][4] = {0};
    __host__ __device__
    Block4() = default;
};

struct Block8
{
    int i,j;
    int blockValue[8][8] = {0};
};
struct BlockComparator4{
    __host__ __device__
    bool operator()(const Block4& block1,const Block4& block2)
    {
        if(block1.i < block2.i) return true;
        else if(block1.i > block2.i) return false;
        else return block1.j < block2.j;
    } 
};

__global__ void matrix_multiplication(Block4* sparse,Block4* transpose,Block4* result,int* prefix_sum_sparse,
    int* num_elements_sparse,int* prefix_sum_transpose,int* num_elements_transpose,int* n_val,int* m_val)
{
    int n = *n_val;
    int m = *m_val;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < n/m && col < n/m)
    {
        Block4 answer;
        long long value[4][4] = {0};
        int pointer1 = 0,pointer2 = 0;
        int start1 = prefix_sum_sparse[row];
        int start2 = prefix_sum_transpose[col];
        while(pointer1 < num_elements_sparse[row] && pointer2 < num_elements_transpose[col])
        {
            if(sparse[start1+pointer1].j == transpose[start2+pointer2].j)
            {
                for(int i = 0;i < 4;i++)
                {
                    for(int j = 0;j < 4;j++)
                    {
                        long long temp = 4294967295;
                        value[i][j] = thrust::min(temp,value[i][j] + sparse[start1+pointer1].blockValue[i][j]*transpose[start2+pointer2].blockValue[i][j]);
                    }
                }
                pointer1++;
                pointer2++;
            }
            else if(sparse[start1+pointer1].j < transpose[start2+pointer2].j) pointer1++;
            else pointer2++;
        }
        bool found = false;
        for(int i =0;i < 4;i++)
        {
            for(int j = 0;j< 4;j++)
            {
                if(value[i][j] > 0) found = true;
                answer.blockValue[i][j] = value[i][j];
            }
        }
        answer.i = row; answer.j = col;
        result[row*(n/m)+col] = answer;
        printf("Row : %d and Col : %d\n",row,col);
    }
}
thrust::pair<thrust::host_vector<Block4>,thrust::host_vector<Block4>> readMatrix4(ifstream& readFile,int n,int m,int k);
// vector<vector<Block8>> readMatrix8(ifstream& readFile,int n,int m,int k);
int main(int argc, char* argv[]) {
    ifstream readfile(argv[1], ios::out | ios::binary);
    if(!readfile) 
    {
        cout << "Cannot open file! " << argv[1] << endl;
        return 1;
    }
    int n=0,m=0,k=0;
    readfile.read((char*) &n,4);
    readfile.read((char*) &m,4);
    readfile.read((char*) &k,4);
    cout << n << " " << m << " " << k << endl;
    thrust::pair<thrust::host_vector<Block4>,thrust::host_vector<Block4>>  matrices = readMatrix4(readfile,n,m,k);
    thrust::host_vector<Block4>& sparse_matrix = matrices.first;
    thrust::host_vector<Block4>& transpose_matrix = matrices.second;
    thrust::host_vector<int> prefix_sum_sparse(n/m,0); // these will be used while accessing the sparse matrix elements
    thrust::host_vector<int> num_elements_sparse(n/m,0);
    thrust::host_vector<int> prefix_sum_transpose(n/m,0);
    thrust::host_vector<int> num_elements_transpose(n/m,0);
    int blockNum = 0;
    while(blockNum < k)
    {
        num_elements_sparse[sparse_matrix[blockNum].i]++;
        num_elements_transpose[transpose_matrix[blockNum].i]++;
        blockNum++;
    }
    for(int i =1;i < n/m;i++)
    {
        prefix_sum_sparse[i] = num_elements_sparse[i-1] + prefix_sum_sparse[i-1];
        prefix_sum_transpose[i] = num_elements_transpose[i-1] + prefix_sum_transpose[i-1];
    }
    thrust::device_vector<Block4> dev_sparse_matrix = sparse_matrix; // device sparse matrix
    thrust::device_vector<Block4> dev_transpose_matrix = transpose_matrix; // device transpose matrix
    thrust::device_vector<int> dev_prefix_sum_sparse = prefix_sum_sparse;
    thrust::device_vector<int> dev_num_elements_sparse = num_elements_sparse;
    thrust::device_vector<int> dev_prefix_sum_transpose = prefix_sum_transpose;
    thrust::device_vector<int> dev_num_elements_transpose = num_elements_transpose;
    /*all device vectors set to perform matrix multiplication operation*/
    /*we'll do matrix multplication using n/m * n/m threads where each calculates some i,j */
    /*need to think about organisation of blocks and threads else everything done */
    int block_x = ((n/m + 15)/16);
    dim3 grid(block_x,block_x,1);
    dim3 block(16,16,1);
    Block4* sparse = thrust::raw_pointer_cast(dev_sparse_matrix.data());
    Block4* transpose = thrust::raw_pointer_cast(dev_transpose_matrix.data());
    Block4* result;
    int* pre_sparse = thrust::raw_pointer_cast(dev_prefix_sum_sparse.data());
    int* num_sparse = thrust::raw_pointer_cast(dev_num_elements_sparse.data());
    int* pre_transpose = thrust::raw_pointer_cast(dev_prefix_sum_transpose.data());
    int* num_transpose = thrust::raw_pointer_cast(dev_num_elements_transpose.data());
    cudaMalloc((void**) &result, (n/m)*(n/m)*sizeof(Block4));
    int host_n = n;
    int host_m = m;
    int* device_n;int* device_m;
    cudaMalloc((void**)&device_n, sizeof(int));
    cudaMalloc((void**)&device_m ,sizeof(int));
    cudaMemcpy(device_n, &host_n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_m,&host_m,sizeof(int),cudaMemcpyHostToDevice);
    matrix_multiplication<<<grid,block>>>(sparse,transpose,result,pre_sparse,num_sparse,pre_transpose,num_transpose,device_n,device_m);
    cudaDeviceSynchronize();
    cout << "finally done \n";
    Block4* final_answer;
    cudaMallocHost((void**)&final_answer,(n/m)*(n/m)*sizeof(Block4));
    cudaMemcpy(final_answer,result, (n/m)*(n/m)*sizeof(Block4), cudaMemcpyDeviceToHost);
    for(int row = 0;row < n/m;row++)
    {
        for(int col= 0;col < n/m;col++)
        {
            if(final_answer[row*(n/m)+col].i == row && final_answer[row*(n/m)+col].j == col)
            {
                for(int i =0;i < 4;i++)
                {
                    for(int j = 0;j < 4;j++)
                    {
                        cout << final_answer[row*(n/m)+col].blockValue[i][j] << " ";
                    }
                    cout << endl;
                }
            }
        }
    }
    cudaFreeHost(final_answer);
    cudaFree(result);
    cudaFree(device_n);
    cudaFree(device_m);
    return 0;
}

thrust::pair<thrust::host_vector<Block4>,thrust::host_vector<Block4>> readMatrix4(ifstream& readFile,int n,int m,int k)
{
    thrust::host_vector<Block4> sparse_matrix(k);
    thrust::host_vector<Block4> transpose_matrix(k);
    int sumValue = 0;
    for(int block=0;block < k;block++)
    {
        int i =0,j=0;
        readFile.read((char*) &i,4);
        readFile.read((char*) &j,4);
        Block4 matrix_block;
        Block4 transpose_block;
        matrix_block.i = i;
        matrix_block.j = j;
        transpose_block.i = j;
        transpose_block.j = i;
        for(int value = 0;value < m*m;value++)
        {
            int number = 0;
            readFile.read((char*) &number,2);
            matrix_block.blockValue[value/m][value%m] = number;
            transpose_block.blockValue[value%m][value/m] = number;
            sumValue += number;
        }
        sparse_matrix[block] = matrix_block;
        transpose_matrix[block] = transpose_block;
    }
    readFile.close();
    // finally we do the sorting of blocks based on their index j for each row
    thrust::sort(sparse_matrix.begin(),sparse_matrix.end(),BlockComparator4());
    thrust::sort(transpose_matrix.begin(),transpose_matrix.end(),BlockComparator4());
    cout << "Sum value is : " << sumValue << endl;
    return thrust::make_pair(sparse_matrix,transpose_matrix);
}

/* here we have in total of N*N threads and each one is evaluating row,col element
 and they do so by for a given thread, in each iteration it loads the common piece of data of 16*16 submatrix
 then they multiply them together to store it in sum then finally update the C array
#include <iostream>

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int N)
{
    // declare shared memory arrays for A and B submatrices
    __shared__ float Asub[16][16];
    __shared__ float Bsub[16][16];

    // calculate global row and column indices for current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize the result for the current thread to zero
    float sum = 0;

    // loop over all submatrices of A and B that contribute to C
    for (int i = 0; i < N / 16; i++)
    {
        // load submatrix of A into shared memory
        Asub[threadIdx.y][threadIdx.x] = A[row * N + (i * 16 + threadIdx.x)];
        // load submatrix of B into shared memory
        Bsub[threadIdx.y][threadIdx.x] = B[(i * 16 + threadIdx.y) * N + col];

        // synchronize threads to ensure all data is loaded into shared memory
        __syncthreads();

        // compute partial result for current thread
        for (int j = 0; j < 16; j++)
        {
            sum += Asub[threadIdx.y][j] * Bsub[j][threadIdx.x];
        }

        // synchronize threads to ensure all data is used before loading new data
        __syncthreads();
    }

    // write the final result to the output matrix
    C[row * N + col] = sum;
}

void matrixMultiply(float* A, float* B, float* C, int N)
{
    // allocate device memory for input and output matrices
    float* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    // copy input matrices from host to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // define block and grid sizes for kernel invocation
    dim3 block(16, 16, 1);
    dim3 grid(N / 16, N / 16, 1);

    // launch kernel
    matrixMultiplyKernel<<<grid, block>>>(d_A, d_B, d_C, N);

    // copy output matrix from device to host
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    const int N = 1024;

    // allocate memory for input and output matrices
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    // initialize input matrices
    for (int i = 0; i < N * N; i++)
    {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // perform matrix multiplication
    matrixMultiply(A, B, C, N);

    // print some elements of the output matrix to verify correctness
    std::cout
*/