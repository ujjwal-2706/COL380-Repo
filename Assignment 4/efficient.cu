#include <iostream>
#include <cuda.h>
#include <thrust/sort.h>
#include <fstream>
using namespace std;

struct Block4
{
    unsigned int i,j;
    unsigned int blockValue[4][4] = {0};
    bool found = false;
    __host__ __device__
    Block4() = default;
};

struct Block8
{
    unsigned int i,j;
    unsigned int blockValue[8][8] = {0};
};
struct BlockComparator4{
    bool operator()(const Block4& block1,const Block4& block2)
    {
        if(block1.i < block2.i) return true;
        else if(block1.i > block2.i) return false;
        else return block1.j < block2.j;
    } 
};

// __global__ 
// void matrix_element_count(Block4* sparse, Block4* transpose,int* prefix_sum_sparse,int* num_elements_sparse,
//     int* prefix_sum_transpose,int* num_elements_transpose,int* n_val,int* m_val,int* counter)
// {
//     int n = *n_val;
//     int m = *m_val;
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int col = blockIdx.y * blockDim.y + threadIdx.y;
//     if(row < n/m && col < n/m)
//     {
//         Block4 answer;
//         unsigned int value[4][4] {0};
//         int pointer1 = 0,pointer2=0;
//         int start1= prefix_sum_sparse[row];
//         int start2= prefix_sum_transpose[col];
//         while(pointer1 < num_elements_sparse[row] && pointer2 < num_elements_transpose[col])
//         {
//             if(sparse[start1+pointer1].j == transpose[start2+pointer2].j)
//             {
//                 for(int i = 0;i < 4;i++)
//                 {
//                     for(int j = 0;j < 4;j++)
//                     {
//                         for(int k =0;k < 4;k++)
//                         {
//                             long long temp = 4294967295;
//                             long long spa = sparse[start1+pointer1].blockValue[i][k];
//                             long long trans = transpose[start2+pointer2].blockValue[j][k];
//                             long long temp2 = value[i][j];
//                             trans = trans*spa + temp2;
//                             value[i][j] = thrust::min(temp,trans);
//                         }
//                     }
//                 }
//                 pointer1++;
//                 pointer2++;
//             }
//             else if(sparse[start1+pointer1].j < transpose[start2+pointer2].j) pointer1++;
//             else pointer2++;
//         }
//         bool found = false;
//         for(int i =0;i < 4;i++)
//         {
//             for(int j = 0;j< 4;j++)
//             {
//                 if(value[i][j] > 0) found = true;
//             }
//         }
//         if(found) addAtomic(counter,1);
//     }
// }
__global__
void matrix_multiplication(Block4* sparse,Block4* transpose,Block4* result,int* prefix_sum_sparse,
    int* num_elements_sparse,int* prefix_sum_transpose,int* num_elements_transpose,int* n_val,int* m_val)
{
    int n = *n_val;
    int m = *m_val;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < n/m && col < n/m)
    {
        Block4 answer;
        unsigned int value[4][4] {0};
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
                        for(int k =0;k < 4;k++)
                        {
                            long long temp = 4294967295;
                            long long spa = sparse[start1+pointer1].blockValue[i][k];
                            long long trans = transpose[start2+pointer2].blockValue[j][k];
                            long long temp2 = value[i][j];
                            trans = trans*spa + temp2;
                            value[i][j] = thrust::min(temp,trans);
                        }
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
            }
        }
        if(found)
        {
            answer.i = row;answer.j = col;
            for(int i =0;i < 15;i++) answer.blockValue[i>>2][i&3] = value[i>>2][i&3];
            answer.found = true;
            printf("Matrix done multiplying\n");
        }
        result[row*(n/m)+col] = answer;
    }
}

void readInput4(ifstream& readFile,Block4* matrix,unsigned int k);
void readInputTranspose4(ifstream& readFile, Block4* matrix,unsigned int k);

int main(int argc, char* argv[]) {

    /*File Reading and parallel device transfer begins*/
    ifstream readfile(argv[1], ios::out | ios::binary);
    if(!readfile) 
    {
        cout << "Cannot open file 1 " << argv[1] << endl;
        return 1;
    }
    unsigned int n1=0,m1=0,k1=0;
    readfile.read((char*) &n1,4);readfile.read((char*) &m1,4);readfile.read((char*) &k1,4);
    cout << n1 << " " << m1 << " " << k1 << endl;
    Block4* matrix1 = (Block4*) malloc(k1*sizeof(Block4));
    readInput4(readfile,matrix1,k1); // file read and ready for transfer

    ifstream readfile2(argv[2],ios::out | ios::binary);
    if(!readfile2)
    {
        cout << "Cannot open file 2 " << argv[2] << endl;
        return 1;
    }
    unsigned int n2=0,m2=0,k2=0;
    readfile2.read((char*) &n2,4);readfile2.read((char*) &m2,4);readfile2.read((char*) &k2,4);
    cout << n2 << " " << m2 << " " << k2 << endl;
    Block4* matrix2 = (Block4*) malloc(k2*sizeof(Block4));
    
    Block4* d_matrix1;
    Block4* d_matrix2;
    cudaMalloc(&d_matrix1,k1*sizeof(Block4));
    cudaMalloc(&d_matrix2,k2*sizeof(Block4));

    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMemcpyAsync(d_matrix1,matrix1,k1*sizeof(Block4),cudaMemcpyHostToDevice,stream1);

    readInputTranspose4(readfile2,matrix2,k2); // file read and ready for transfer

    cudaMemcpyAsync(d_matrix2,matrix2,k2*sizeof(Block4),cudaMemcpyHostToDevice,stream2);

    // these will be used while accessing the sparse matrix elements
    int prefix_sum_sparse[n1/m1] {0}; 
    int num_elements_sparse[n1/m1] {0};
    int prefix_sum_transpose[n2/m2] {0};
    int num_elements_transpose[n2/m2] {0};

    int* d_prefix_sum_sparse;int* d_num_elements_sparse;int* d_prefix_sum_transpose;int* d_num_elements_transpose;
    cudaMalloc(&d_prefix_sum_sparse,(n1/m1)*sizeof(int));
    cudaMalloc(&d_num_elements_sparse,(n1/m1)*sizeof(int));
    cudaMalloc(&d_prefix_sum_transpose,(n2/m2)*sizeof(int));
    cudaMalloc(&d_num_elements_transpose,(n2/m2)*sizeof(int));
    int n = n1,m = m1;
    for(int block = 0;block < k1; block++)
    {
        num_elements_sparse[matrix1[block].i]++;
    }
    cudaMemcpyAsync(d_num_elements_sparse,num_elements_sparse,(n1/m1)*sizeof(int),cudaMemcpyHostToDevice,stream1);
    for(int block=0;block<k2;block++)
    {
        num_elements_transpose[matrix2[block].i]++;
    }
    cudaMemcpyAsync(d_num_elements_transpose,num_elements_transpose,(n2/m2)*sizeof(int),cudaMemcpyHostToDevice,stream2);
    for(int i =1;i < n/m;i++)
    {
        prefix_sum_sparse[i] = num_elements_sparse[i-1] + prefix_sum_sparse[i-1];
        prefix_sum_transpose[i] = num_elements_transpose[i-1] + prefix_sum_transpose[i-1];
    }
    cudaMemcpyAsync(d_prefix_sum_sparse,prefix_sum_sparse,(n1/m1)*sizeof(int),cudaMemcpyHostToDevice,stream1);
    cudaMemcpyAsync(d_prefix_sum_transpose,prefix_sum_transpose,(n2/m2)*sizeof(int),cudaMemcpyHostToDevice,stream2);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    /*Data sending to device done now we launch kernel*/

    int block_x = ((n/m + 15)/16);
    dim3 grid(block_x,block_x,1);
    dim3 block(16,16,1);
    /*kernel grid and blocks set*/

    // int* counter;
    // cudaMalloc((void**) &counter,sizeof(int));
    // int count = 0;
    // cudaMemcpy(counter,&count,sizeof(int),cudaMemcpyHostToDevice);
    int host_n = n;
    int host_m = m;
    int* device_n;int* device_m;
    cudaMalloc((void**)&device_n, sizeof(int));
    cudaMalloc((void**)&device_m ,sizeof(int));
    cudaMemcpy(device_n, &host_n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_m,&host_m,sizeof(int),cudaMemcpyHostToDevice);
    // matrix_element_count(d_matrix1,d_matrix2,d_prefix_sum_sparse,d_num_elements_sparse,d_prefix_sum_transpose,
    //     d_num_elements_transpose,device_n,device_m,counter);
    // cudaMemcpy(&count,counter,sizeof(int),cudaMemcpyDeviceToHost);
    Block4* result;
    cudaMalloc((void**) &result, (n/m)*(n/m)*sizeof(Block4));
    // int count2 = count;
    // count = 0;
    // cudaMemcpy(&count,counter,sizeof(int),cudaMemcpyDeviceToHost);
    matrix_multiplication<<<grid,block>>>(d_matrix1,d_matrix2,result,d_prefix_sum_sparse,d_num_elements_sparse
        ,d_prefix_sum_transpose,d_num_elements_transpose,device_n,device_m);
    cout << "finally done \n";
    Block4* final_answer;
    final_answer = (Block4*) malloc((n/m)*(n/m)*sizeof(Block4));
    cudaMemcpy(final_answer,result, (n/m)*(n/m)*sizeof(Block4), cudaMemcpyDeviceToHost);
    int total_elements = 0;
    for(int row = 0;row < (n/m);row++)
    {
        for(int col =0;col < (n/m);col++)
        {
            if(final_answer[row*(n/m)+col].found)
            {
                total_elements++;
                for(int i = 0;i < 4;i++)
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
    cout << total_elements << endl;
    free(final_answer);
    cudaFree(result);
    cudaFree(device_n);
    cudaFree(device_m);

    free(matrix1);
    free(matrix2);
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_prefix_sum_sparse);
    cudaFree(d_num_elements_sparse);
    cudaFree(d_prefix_sum_transpose);
    cudaFree(d_num_elements_transpose);
    // cudaFree(counter);
    return 0;
}

void readInput4(ifstream& readFile,Block4* matrix,unsigned int k)
{
    for(int block=0;block < k;block++)
    {
        unsigned int i =0,j=0;
        readFile.read((char*) &i,4);
        readFile.read((char*) &j,4);
        Block4 matrix_block;
        matrix_block.i = i;
        matrix_block.j = j;
        for(int value = 0;value < 4*4;value++)
        {
            unsigned int number = 0;
            readFile.read((char*) &number,2);
            matrix_block.blockValue[value>>2][value & 3] = number;
        }
        matrix[block] = matrix_block;
    }
    readFile.close();
    thrust::sort(matrix,matrix+k,BlockComparator4());
}

void readInputTranspose4(ifstream& readFile, Block4* matrix,unsigned int k)
{
    for(int block=0;block < k;block++)
    {
        unsigned int i =0,j=0;
        readFile.read((char*) &i,4);
        readFile.read((char*) &j,4);
        Block4 matrix_block;
        matrix_block.i = j;
        matrix_block.j = i;
        for(int value = 0;value < 4*4;value++)
        {
            unsigned int number = 0;
            readFile.read((char*) &number,2);
            matrix_block.blockValue[value & 3][value >> 2] = number;
        }
        matrix[block] = matrix_block;
    }
    readFile.close();
    thrust::sort(matrix,matrix+k,BlockComparator4());
}

/* here we have in total of N*N threads and each one is evaluating row,col element
 and they do so by for a given thread, in each iteration it loads the common piece of data of 16*16 submatrix
 then they multiply them together to store it in sum then finally update the C array
*/
/*Can save memory from short instead of unsigned int */

/*Can assign full memory in GPU but only transfer partial in CPU */