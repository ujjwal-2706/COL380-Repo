#include <iostream>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <fstream>
#include <vector>
using namespace std;

struct Block4
{
    bool found = false;
    unsigned int i,j;
    unsigned int blockValue[16] = {0};
    __host__ __device__
    Block4() = default;
};
struct Block8
{
    bool found = false;
    unsigned int i,j;
    unsigned int blockValue[64] = {0};
    __host__ __device__
    Block8() = default;
};
struct BlockComparator4{
    bool operator()(const Block4& block1,const Block4& block2)
    {
        if(block1.i < block2.i) return true;
        else if(block1.i > block2.i) return false;
        else return block1.j < block2.j;
    } 
};
struct BlockComparator8{
    bool operator()(const Block8& block1,const Block8& block2)
    {
        if(block1.i < block2.i) return true;
        else if(block1.i > block2.i) return false;
        else return block1.j < block2.j;
    } 
};

__global__
void matrix_multiplication4(Block4* sparse,Block4* transpose,Block4* result,int* prefix_sum_sparse,
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
                            unsigned long long temp = 4294967295;
                            unsigned long long spa = sparse[start1+pointer1].blockValue[4*i+k];
                            unsigned long long trans = transpose[start2+pointer2].blockValue[4*j+k];
                            unsigned long long temp2 = value[i][j];
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
            for(int i =0;i < 16;i++) answer.blockValue[i] = value[i>>2][i&3];
            answer.found = true;
        }
        result[row*(n/m)+col] = answer;
    }
}

__global__
void matrix_multiplication8(Block8* sparse,Block8* transpose,Block8* result,int* prefix_sum_sparse,
    int* num_elements_sparse,int* prefix_sum_transpose,int* num_elements_transpose,int* n_val,int* m_val)
{
    int n = *n_val;
    int m = *m_val;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < n/m && col < n/m)
    {
        Block8 answer;
        unsigned int value[8][8] {0};
        int pointer1 = 0,pointer2 = 0;
        int start1 = prefix_sum_sparse[row];
        int start2 = prefix_sum_transpose[col];
        while(pointer1 < num_elements_sparse[row] && pointer2 < num_elements_transpose[col])
        {
            if(sparse[start1+pointer1].j == transpose[start2+pointer2].j)
            {
                for(int i = 0;i < 8;i++)
                {
                    for(int j = 0;j < 8;j++)
                    {
                        for(int k =0;k < 8;k++)
                        {
                            unsigned long long temp = 4294967295;
                            unsigned long long spa = sparse[start1+pointer1].blockValue[8*i+k];
                            unsigned long long trans = transpose[start2+pointer2].blockValue[8*j+k];
                            unsigned long long temp2 = value[i][j];
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
        for(int i =0;i < 8;i++)
        {
            for(int j = 0;j< 8;j++)
            {
                if(value[i][j] > 0) found = true;
            }
        }
        if(found)
        {
            answer.i = row;answer.j = col;
            for(int i =0;i < 64;i++) answer.blockValue[i] = value[i>>3][i&7];
            answer.found = true;
        }
        result[row*(n/m)+col] = answer;
    }
}

void readInput4(ifstream& readFile,Block4* matrix,unsigned int k);
void readInput8(ifstream& readFile,Block8* matrix,unsigned int k);
void readInputTranspose4(ifstream& readFile, Block4* matrix,unsigned int k);
void readInputTranspose8(ifstream& readFile, Block8* matrix,unsigned int k);
void writeFile4(vector<Block4>& blocks_output,unsigned int k,unsigned int n,unsigned int m,string output_file);
void writeFile8(vector<Block8>& blocks_output,unsigned int k,unsigned int n,unsigned int m,string output_file);

int main(int argc, char* argv[]) {

    /*File Reading and parallel device transfer begins*/
    cout << "Size of Block4 is : " << sizeof(Block4) << endl;
    cout << "Size of Block8 is : " << sizeof(Block8) << endl;
    ifstream readfile(argv[1], ios::out | ios::binary);
    if(!readfile) 
    {
        cout << "Cannot open file 1 " << argv[1] << endl;
        return 1;
    }
    unsigned int n1=0,m1=0,k1=0;
    readfile.read((char*) &n1,4);readfile.read((char*) &m1,4);readfile.read((char*) &k1,4);
    cout << n1 << " " << m1 << " " << k1 << endl;
    if(m1 == 4 )
    {
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

        int host_n = n;
        int host_m = m;
        int* device_n;int* device_m;
        cudaMalloc((void**)&device_n, sizeof(int));
        cudaMalloc((void**)&device_m ,sizeof(int));
        cudaMemcpy(device_n, &host_n, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(device_m,&host_m,sizeof(int),cudaMemcpyHostToDevice);

        Block4* result;
        cudaMalloc((void**) &result, (n/m)*(n/m)*sizeof(Block4));
    
        matrix_multiplication4<<<grid,block>>>(d_matrix1,d_matrix2,result,d_prefix_sum_sparse,d_num_elements_sparse
            ,d_prefix_sum_transpose,d_num_elements_transpose,device_n,device_m);
        Block4* final_answer;
        final_answer = (Block4*) malloc((n/m)*(n/m)*sizeof(Block4));
        cudaMemcpy(final_answer,result, (n/m)*(n/m)*sizeof(Block4), cudaMemcpyDeviceToHost);
        cout << "finally done \n";
        unsigned int total_elements = 0;
        printf("%u \n",final_answer[0].blockValue[15]);
        vector<Block4> blocks_output;
        for(int row = 0;row < (n/m);row++)
        {
            for(int col =0;col < (n/m);col++)
            {
                if(final_answer[row*(n/m)+col].found)
                {
                    total_elements++;
                    blocks_output.push_back(final_answer[row*(n/m)+col]);
                }
            }
        }
        cout << total_elements << endl;
        string output_name = argv[3];
        writeFile4(blocks_output,total_elements,n,m,output_name);
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
    }
    else
    {
        Block8* matrix1 = (Block8*) malloc(k1*sizeof(Block8));
        readInput8(readfile,matrix1,k1); // file read and ready for transfer

        ifstream readfile2(argv[2],ios::out | ios::binary);
        if(!readfile2)
        {
            cout << "Cannot open file 2 " << argv[2] << endl;
            return 1;
        }
        unsigned int n2=0,m2=0,k2=0;
        readfile2.read((char*) &n2,4);readfile2.read((char*) &m2,4);readfile2.read((char*) &k2,4);
        cout << n2 << " " << m2 << " " << k2 << endl;
        Block8* matrix2 = (Block8*) malloc(k2*sizeof(Block8));
        
        Block8* d_matrix1;
        Block8* d_matrix2;
        cudaMalloc(&d_matrix1,k1*sizeof(Block8));
        cudaMalloc(&d_matrix2,k2*sizeof(Block8));

        cudaStream_t stream1,stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        cudaMemcpyAsync(d_matrix1,matrix1,k1*sizeof(Block8),cudaMemcpyHostToDevice,stream1);

        readInputTranspose8(readfile2,matrix2,k2); // file read and ready for transfer

        cudaMemcpyAsync(d_matrix2,matrix2,k2*sizeof(Block8),cudaMemcpyHostToDevice,stream2);

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

        int host_n = n;
        int host_m = m;
        int* device_n;int* device_m;
        cudaMalloc((void**)&device_n, sizeof(int));
        cudaMalloc((void**)&device_m ,sizeof(int));
        cudaMemcpy(device_n, &host_n, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(device_m,&host_m,sizeof(int),cudaMemcpyHostToDevice);

        Block8* result;
        cudaMalloc((void**) &result, (n/m)*(n/m)*sizeof(Block8));
    
        matrix_multiplication8<<<grid,block>>>(d_matrix1,d_matrix2,result,d_prefix_sum_sparse,d_num_elements_sparse
            ,d_prefix_sum_transpose,d_num_elements_transpose,device_n,device_m);
        Block8* final_answer;
        final_answer = (Block8*) malloc((n/m)*(n/m)*sizeof(Block8));
        cudaMemcpy(final_answer,result, (n/m)*(n/m)*sizeof(Block8), cudaMemcpyDeviceToHost);
        cout << "finally done \n";
        unsigned int total_elements = 0;
        printf("%u \n",final_answer[0].blockValue[15]);
        vector<Block8> blocks_output;
        for(int row = 0;row < (n/m);row++)
        {
            for(int col =0;col < (n/m);col++)
            {
                if(final_answer[row*(n/m)+col].found)
                {
                    total_elements++;
                    blocks_output.push_back(final_answer[row*(n/m)+col]);
                }
            }
        }
        cout << total_elements << endl;
        string output_name = argv[3];
        writeFile8(blocks_output,total_elements,n,m,output_name);
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
    }
    
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
            matrix_block.blockValue[value] = number;
        }
        matrix[block] = matrix_block;
    }
    readFile.close();
    thrust::sort(matrix,matrix+k,BlockComparator4());
}

void readInput8(ifstream& readFile,Block8* matrix,unsigned int k)
{
    for(int block=0;block < k;block++)
    {
        unsigned int i =0,j=0;
        readFile.read((char*) &i,4);
        readFile.read((char*) &j,4);
        Block8 matrix_block;
        matrix_block.i = i;
        matrix_block.j = j;
        for(int value = 0;value < 8*8;value++)
        {
            unsigned int number = 0;
            readFile.read((char*) &number,2);
            matrix_block.blockValue[value] = number;
        }
        matrix[block] = matrix_block;
    }
    readFile.close();
    thrust::sort(matrix,matrix+k,BlockComparator8());
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
            matrix_block.blockValue[(value & 3)*4 + (value >> 2)] = number;
        }
        matrix[block] = matrix_block;
    }
    readFile.close();
    thrust::sort(matrix,matrix+k,BlockComparator4());
}

void readInputTranspose8(ifstream& readFile, Block8* matrix,unsigned int k)
{
    for(int block=0;block < k;block++)
    {
        unsigned int i =0,j=0;
        readFile.read((char*) &i,4);
        readFile.read((char*) &j,4);
        Block8 matrix_block;
        matrix_block.i = j;
        matrix_block.j = i;
        for(int value = 0;value < 8*8;value++)
        {
            unsigned int number = 0;
            readFile.read((char*) &number,2);
            matrix_block.blockValue[(value & 7)*8 + (value >> 3)] = number;
        }
        matrix[block] = matrix_block;
    }
    readFile.close();
    thrust::sort(matrix,matrix+k,BlockComparator8());
}

void writeFile4(vector<Block4>& blocks_output,unsigned int k,unsigned int n,unsigned int m,string output_file)
{
    ofstream writefile(output_file,ios::out|ios::binary);
    if(!writefile)
    {
        cout << "Cannot open write file!" << endl;
    }
    writefile.write((char*) &n,4);
    writefile.write((char*) &m,4);
    writefile.write((char*) &k,4);
    for(int block = 0;block < blocks_output.size();block++)
    {
        Block4& elements = blocks_output[block];
        unsigned int i = elements.i;
        unsigned int j = elements.j;
        writefile.write((char*) &i,4);
        writefile.write((char*) &j,4);
        for(int val = 0;val < 16;val++)
        {
            unsigned int value = elements.blockValue[val];
            writefile.write((char*) &value,4);
        }
    }
    writefile.close();
}

void writeFile8(vector<Block8>& blocks_output,unsigned int k,unsigned int n,unsigned int m,string output_file)
{
    ofstream writefile(output_file,ios::out|ios::binary);
    if(!writefile)
    {
        cout << "Cannot open write file!" << endl;
    }
    writefile.write((char*) &n,4);
    writefile.write((char*) &m,4);
    writefile.write((char*) &k,4);
    for(int block = 0;block < blocks_output.size();block++)
    {
        Block8& elements = blocks_output[block];
        unsigned int i = elements.i;
        unsigned int j = elements.j;
        writefile.write((char*) &i,4);
        writefile.write((char*) &j,4);
        for(int val = 0;val < 64;val++)
        {
            unsigned int value = elements.blockValue[val];
            writefile.write((char*) &value,4);
        }
    }
    writefile.close();
}
/* here we have in total of N*N threads and each one is evaluating row,col element
 and they do so by for a given thread, in each iteration it loads the common piece of data of 16*16 submatrix
 then they multiply them together to store it in sum then finally update the C array
*/
/*Can save memory from short instead of unsigned int */

/*Can assign full memory in GPU but only transfer partial in CPU */


// Value changing for 2^32-1 on copy back!!!!!!!!!!!!!!!!
