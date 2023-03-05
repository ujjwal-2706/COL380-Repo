#include <mpi.h>
#include <iostream>
#include <fstream>
#include <bits/stdc++.h>
using namespace std;

string INPUT = "./TestCases/test0/test-input-0.gra";
// we send node i data to i%(size-1) + 1 process
int main(int argc,char* argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank == 0) // coordinator process
    {
        // first it will read the graph and distribute it among other cluster nodes
        ifstream input(INPUT);
        if(!input) cout << "Unable to Open File!\n"; 
        int m=0,n=0;
        input.read((char*) &n,4);
        input.read((char*) &m,4);
        vector<int> degrees(n,-1); //we get the degree of ith node
        vector<vector<int>> graph(n,vector<int>());
        cout << n << " " << m << endl;
        for(int i =0;i < n;i++)
        {
            int node=0;int degree = 0;
            input.read((char*) &node,4);
            input.read((char*) &degree,4);
            degrees[node] = degree;
            for(int edge = 0; edge<degree;edge++)
            {
                int neighbour = 0;
                input.read((char*) &neighbour,4);
                graph[node].push_back(neighbour); // as given in neighbours form
            }
        }
        input.close();
        // now we need to do pre processing to send the data about edges to different processes
        // distribute vertices equally among the other processes 1 to size-1.
        // then just coordinate requests and finally stop on receiving complete signal
        // we send node i data to i%(size-1) + 1 process and the tag will be rank of process
        vector<int> verticesSent(size-1,0); // how many vertices will be sent to each process
        vector<int> edgeCount(n,0); // for each vertex what is the edge count
        for(int i =0;i < n;i++)
        {
            int countEdges = 0;
            verticesSent[i%(size-1)]++;
            for(int edge = 0;edge < graph[i].size();edge++)
            {
                int neighbour = graph[i][edge];
                if((degrees[i] < degrees[neighbour])|| (degrees[i] == degrees[neighbour] && i < neighbour))
                {
                    countEdges++;
                }
            }
            edgeCount[i] = countEdges;
        }
        // now we send the vertices
        for(int i =0;i < size-1;i++)
        {
            MPI_Send(&verticesSent[i],1,MPI_INT,i+1,i+1,MPI_COMM_WORLD); // send number of vertices to process i+1
        }
        for(int i =0;i < n;i++)
        {
           // send vertex number first and then number of edges then the edges to be sent
            int process = i%(size-1)+1;
            MPI_Send(&i,1,MPI_INT,process,process,MPI_COMM_WORLD);
            MPI_Send(&edgeCount[i],1,MPI_INT,process,process,MPI_COMM_WORLD);
            for(int edge = 0;edge < graph[i].size();edge++)
            {
                int neighbour = graph[i][edge];
                if((degrees[i] < degrees[neighbour])|| (degrees[i] == degrees[neighbour] && i < neighbour))
                {
                    MPI_Send(&neighbour,1,MPI_INT,process,process,MPI_COMM_WORLD); // send the neighbour
                }
            }
        }
    }
    else // other processes
    {
        // first they receive the graph
        // then we do iterations
        // then we send complete signals and stop
        int vertices = 0;
        MPI_Status status;
        MPI_Recv(&vertices,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status); // received total number of vertices
        vector<vector<int>> subgraph(vertices,vector<int>());
        int edgeNumbers = 0;
        for(int vertex = 0;vertex < vertices;vertex++)
        {
            int vertexNumber =0,edgeCount = 0;
            MPI_Recv(&vertexNumber,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
            MPI_Recv(&edgeCount,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
            for(int edge = 0;edge < edgeCount ;edge++)
            {
                int neighbour = -1;
                MPI_Recv(&neighbour,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
                subgraph[vertex].push_back(neighbour);
                edgeNumbers++;
            }
        }
        // graph distribution done
        cout << "Vertices Received : "<< vertices << endl;
        cout << "Edges Received : " << edgeNumbers << endl;
        // now initialize trussities and start with the iterations
    }
    MPI_Finalize();
    return 0;
}