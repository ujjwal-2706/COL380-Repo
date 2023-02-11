#include<iostream>
#include<fstream>
#include<chrono>
#include <vector>
#include<algorithm>
#include<omp.h>
#include "library.hpp"
using namespace std;
// here we do result[i][k] = Outer(result[i][k],Inner(matrix[i][j],matrix[i][k])) with blocks
void blockMultiplication(int result[],int matrix[],int& n, int& m,int blockValues[],int i,int j,int k)
{
    if(blockValues[i*(n/m) + j] >0 && blockValues[j*(n/m)+k]>0)
    {
        for(int row = 0;row < m;row++)
        {
            for(int col = 0;col < m;col++)
            {
                for(int imm =0;imm < m;imm++)
                {
                    result[(i*m+row)*n + k*m + col] = Outer(result[(i*m+row)*n + k*m + col],Inner(matrix[(i*m+row)*n+j*m+imm],matrix[(j*m+imm)*n+k*m+col]));
                    result[(i*m+row)*n + k*m + col] = min(result[(i*m+row)*n + k*m + col],(1<<16)); 

                }
            }
        }
    }
}

class Block{
    public:
        vector<vector<int>> block;
        int i,j;
        Block() = default;
        Block(int m,int i ,int j)
        {
            block = vector<vector<int>> (m,vector<int>(m,0));
            this->i = i;
            this->j = j;
        }
};

struct BlockComparator{
    bool operator()(const Block& block1,const Block& block2)
    {
        return block1.j < block2.j;
    } 
};

// fill symmetric sparse matrix completely in sorted order of j for matrix
// only fill i <=k entries in answer;
vector<vector<Block>> sparseMultiplication(vector<vector<Block>>& matrix,int n,int m)
{
    vector<vector<Block>> answer(n/m,vector<Block>());
    #pragma omp parallel for
    for(int i =0;i < n/m;i++)
    {
        for(int k=i;k < n/m;k++)
        {
            bool found = false;
            int pointer1 = 0;
            int pointer2 = 0;
            Block temp(m,i,k);
            while(pointer1 < matrix[i].size() && pointer2 < matrix[k].size())
            {
                if(matrix[i][pointer1].j == matrix[k][pointer2].j)
                {
                    found = true;
                    for(int row =0;row < m;row++)
                    {
                        for(int col =0;col < m;col++)
                        {
                            for(int imm =0;imm < m;imm++)
                            {
                                temp.block[row][col] = Outer(temp.block[row][col],Inner(matrix[i][pointer1].block[row][imm],matrix[k][pointer2].block[imm][col]));
                                temp.block[row][col] = min(temp.block[row][col],(1<<16));
                            }
                        }
                    }
                    pointer1++;
                    pointer2++;
                }
                else if(matrix[i][pointer1].j < matrix[k][pointer2].j)
                {
                    pointer1++;
                }
                else
                {
                    pointer2++;
                }
            }
            if(found)
            {
                answer[i].push_back(temp);
            }
            cout << "Block " <<i << " " << k << "done!" << endl;
        }
    }
    return answer;
}
// we will store a vector of vectors for the sparse matrix
// for matrix multiplication we find first the block i,k value by using symmetry as i,j and k,j multiplication and addition
// for each i and k, we move and checking where both non zero j and then we multiply and update i,k block of final matrix
// finally after all this we get k value as i<= k entries of vector of vectors
int main(int argc,char* argv[])
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    ifstream rf("input2", ios::out | ios::binary);
    if(!rf) 
    {
        cout << "Cannot open file!" << endl;
        return 1;
    }
    int n,m,k;
    rf.read((char*) &n,4);
    rf.read((char*) &m,4);
    rf.read((char*) &k,4);
    cout << n << " " << m<< " "<<k<<  endl;
    // int matrix[n*n]= {};
    // int blockValues[(n/m) * (n/m)] = {}; // 0 means all block is zero else 1
    vector<vector<Block>> matrix(n/m,vector<Block>());
    for(int blockNum = 0;blockNum < k;blockNum++)
    {
        int i,j;
        rf.read((char*) &i,4);
        rf.read((char*) &j,4);
        // blockValues[i*(n/m) + j] = 1;
        // blockValues[j*(n/m) + i] = 1;
        Block lower(m,i,j);
        Block upper(m,j,i);
        // i *= m;
        // j *= m;
        for(int row = 0;row < m;row++)
        {
            for(int col =0;col < m;col++)
            {
                // rf.read((char*) &matrix[(i+row)*n + j +col],1);
                rf.read((char*) &lower.block[row][col],1);
                // matrix[(j+col)*n + (i+row)] = matrix[(i+row)*n + j + col];
                upper.block[col][row] = lower.block[row][col];
            }
        }
        matrix[i].push_back(lower);
        matrix[j].push_back(upper);
    }
    rf.close();
    for(int i =0;i < n/m;i++)
    {
        sort(matrix[i].begin(),matrix[i].end(),BlockComparator());
    }
    cout << "Matrix initialzation in Block format done!" << endl;
    // int result[n*n] = {};
    // for(int i = 0;i < (n/m);i++)
    // {
    //     for(int k =0;k < (n/m);k++)
    //     {
    //         for(int j =0;j < (n/m);j++)
    //         {
    //             blockMultiplication(result,matrix,n,m,blockValues,i,j,k);
    //         }
    //     }
    // }

    // for(int i =0;i < n;i++)
    // {
    //     for(int k =0;k < n;k++)
    //     {
    //         for(int j =0;j < n;j++)
    //         {
    //             result[i*n + k] = Outer(result[i*n+k],Inner(matrix[i*n+j],matrix[j*n+k]));
    //             result[i*n+k] = min(result[i*n+k],(1<<16));
    //         }
    //     }
    // }
    // for(int i =0;i < n;i++)
    // {
    //     for(int j =0;j < n;j++)
    //     {
    //         cout << result[i*n+j] << " ";
    //     }
    //     cout << endl;
    // }

    vector<vector<Block>> answer = sparseMultiplication(matrix,n,m);
    int nonZeroBlocks = 0;
    for(int i =0;i < n/m;i++)
    {
        nonZeroBlocks += answer[i].size();
    }
    cout << "Value of Our k is : " << nonZeroBlocks << endl;
    ifstream rf2("output2",ios::out | ios::binary);
    if(!rf2) 
    {
        cout << "Cannot open file!" << endl;
        return 1;
    }
    int n1,m1,k1;
    rf2.read((char*) &n1,4);
    rf2.read((char*) &m1,4);
    rf2.read((char*) &k1,4);
    // int matrix1[n1*n1] = {};
    // for(int block = 0;block < k1;block++)
    // {
    //     int i1,j1;
    //     rf2.read((char*) &i1,4);
    //     rf2.read((char*) &j1,4);
    //     // blockValues[i*(n/m) + j] = 1;
    //     i1 *= m;
    //     j1 *= m;
    //     for(int row = 0;row < m;row++)
    //     {
    //         for(int col =0;col < m;col++)
    //         {
    //             // cout << i1 + row << " " << j1 + col << endl;
    //             rf2.read((char*) &matrix1[(i1+row)*n + j1 +col],2);
    //             matrix1[(j1+col)*n + i1 + row] = matrix1[(i1+row)*n + j1 + col];
    //         }
    //     }
    // }
    rf2.close();
    // for(int i =0;i < n;i++)
    // {
    //     for(int j =0;j < n;j++)
    //     {
    //         // cout << matrix1[i*n+j] << " ";
    //         if(matrix1[i*n+j] != result[i*n+j])
    //         {
    //             cout << "not equal at : " << i << " " << j << endl;
    //         }
    //     }
    //     // cout << endl;
    // }
    // cout << n1 <<" "<< m1<< " " << k1 << endl;
    // cout << n << " " << m<< " "<<k<<  endl;
    // cout << "file read successfully\n";
    cout << "Value of Original k is : " << k1 << endl;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    cout << "Finished in : " << elapsed_seconds.count() << endl;
    return 0;
}