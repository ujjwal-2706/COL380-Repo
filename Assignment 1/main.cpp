#include<iostream>
#include<fstream>
#include<chrono>
#include<vector>
#include<algorithm>
#include<omp.h>
#include "library.hpp"
using namespace std;

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

struct MatrixDimension{
    vector<vector<Block>> matrix;
    int n,m,k;
};
MatrixDimension readBinaryFile(char* fileName,int bytesize);
vector<vector<Block>> sparseMultiplication(vector<vector<Block>>& matrix,int n,int m);
bool compareResults(vector<vector<Block>> result,vector<vector<Block>> original,int n,int m);

int main(int argc,char* argv[])
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    MatrixDimension inputfullMatrix = readBinaryFile(argv[1],1);
    cout << "Matrix initialzation in Block format done!" << endl;

    vector<vector<Block>> answer = sparseMultiplication(inputfullMatrix.matrix,inputfullMatrix.n,inputfullMatrix.m);
    int nonZeroBlocks = 0;
    for(int i =0;i < inputfullMatrix.n/inputfullMatrix.m;i++)
    {
        nonZeroBlocks += answer[i].size();
    }
    cout << "Value of Our k is : " << nonZeroBlocks << endl;
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    cout << "Processing time is : " << elapsed_seconds.count() << endl;
    MatrixDimension outputfullMatrix = readBinaryFile(argv[2],2);
    cout << "Value of Original k is : " << outputfullMatrix.k << endl;
    bool check = compareResults(answer,outputfullMatrix.matrix,outputfullMatrix.n,outputfullMatrix.m);
    if(check)
    {
        cout << "Test Passed!" << endl;
    }
    else
    {
        cout << "Something's wrong" << endl;
    }
    
    return 0;
}


MatrixDimension readBinaryFile(char* fileName,int bytesize)
{
    ifstream readfile(fileName, ios::out | ios::binary);
    MatrixDimension answer;
    if(!readfile) 
    {
        cout << "Cannot open file! " << fileName << endl;
        return answer;
    }
    int n,m,k;
    readfile.read((char*) &n,4);
    readfile.read((char*) &m,4);
    readfile.read((char*) &k,4);
    answer.n = n;
    answer.k = k;
    answer.m = m;
    vector<vector<Block>> matrix(n/m,vector<Block>());
    for(int blockNum = 0;blockNum < k;blockNum++)
    {
        int i,j;
        readfile.read((char*) &i,4);
        readfile.read((char*) &j,4);
        Block lower(m,i,j);
        Block upper(m,j,i);
        for(int row = 0;row < m;row++)
        {
            for(int col =0;col < m;col++)
            {
                readfile.read((char*) &lower.block[row][col],bytesize);
                upper.block[col][row] = lower.block[row][col];
            }
        }
        if(i != j)
        {
            matrix[i].push_back(lower);
            matrix[j].push_back(upper);
        }
        else
        {
            matrix[i].push_back(lower);
        }
    }
    readfile.close();
    for(int i =0;i < n/m;i++)
    {
        sort(matrix[i].begin(),matrix[i].end(),BlockComparator());
    }
    answer.matrix = matrix;
    return answer;
}
// fill symmetric sparse matrix completely in sorted order of j for matrix
// only fill i <=k entries in answer;
// we will store a vector of vectors for the sparse matrix
// for matrix multiplication we find first the block i,k value by using symmetry as i,j and k,j multiplication and addition
// for each i and k, we move and checking where both non zero j and then we multiply and update i,k block of final matrix
// finally after all this we get k value as i<= k entries of vector of vectors
vector<vector<Block>> sparseMultiplication(vector<vector<Block>>& matrix,int n,int m)
{
    vector<vector<Block>> answer(n/m,vector<Block>());
    // #pragma omp parallel for num_threads(128)
    omp_set_num_threads(128);
    #pragma omp parallel
    {
        #pragma omp single
        {
            for(int val =0;val < n/m;val++)
            {
                int i = val;
                #pragma omp task
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
                                            temp.block[row][col] = Outer(temp.block[row][col],Inner(matrix[i][pointer1].block[row][imm],matrix[k][pointer2].block[col][imm]));
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
                    }
                }
            }
        }
    }
    return answer;
}

bool compareResults(vector<vector<Block>> result,vector<vector<Block>> original,int n,int m)
{
    for(int i = 0;i < n/m;i++)
    {
        int pointer = 0;
        while(pointer < original[i].size() && original[i][pointer].j < i)
        {
            pointer++;
        }
        int pointer1 = 0;
        while(pointer1 < result[i].size() && pointer < original[i].size())
        {
            if(result[i][pointer1].j != original[i][pointer].j)
            {
                return false;
            }
            for(int row = 0;row < m;row++)
            {
                for(int col = 0;col < m;col++)
                {
                    if(result[i][pointer1].block[row][col] != original[i][pointer].block[row][col])
                    {
                        return false;
                    }
                }
            }
            pointer++;
            pointer1++;
        }
        if(!(pointer == original[i].size() && pointer1 == result[i].size()))
        {
            return false;
        }
    }
    return true;
}