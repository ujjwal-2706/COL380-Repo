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
        // we will send degrees list to each process initially too
        for(int i =0;i < size-1;i++)
        {
            MPI_Send(&n,1,MPI_INT,i+1,i+1,MPI_COMM_WORLD);
            MPI_Send(&degrees[0],n,MPI_INT,i+1,i+1,MPI_COMM_WORLD);
        }

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
            int neighbours[edgeCount[i]];
            int count = 0;
            for(int edge = 0;edge < graph[i].size();edge++)
            {
                int neighbour = graph[i][edge];
                if((degrees[i] < degrees[neighbour])|| (degrees[i] == degrees[neighbour] && i < neighbour))
                {
                    neighbours[count] = neighbour;
                    count++;
                }
            }
            MPI_Send(neighbours,edgeCount[i],MPI_INT,process,process,MPI_COMM_WORLD); // send all edges simultaneously 
        }
        // now we will do the triangle enumeration loop by storing the count table from each process 1 to size-1
        int triangleEnumCount[size][size-1] {}; // ij element means that ith process wants to send j+1 th process pij elements
        MPI_Status status;
        for(int i =0;i < size-1;i++)
        {
            MPI_Recv(triangleEnumCount[i+1],size-1,MPI_INT,i+1,i+1,MPI_COMM_WORLD,&status);
        }
        // triangle count received
        cout << "Triangle Count Received!\n";
        for(int j =1;j<size;j++)
        {
            int process_j[size-1] {};
            for(int i =1;i<size;i++)
            {
                process_j[i-1] = triangleEnumCount[i][j-1];
            }
            MPI_Send(process_j,size-1,MPI_INT,j,j,MPI_COMM_WORLD); // all data of process j sent to process j
        }
        cout << "Sending done by process 0\n";
    }
    else // other processes
    {
        int n=0;
        int vertices = 0;
        MPI_Status status;
        MPI_Recv(&n,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
        vector<int> degrees(n,-1);
        MPI_Recv(&degrees[0],n,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
        MPI_Recv(&vertices,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status); // received total number of vertices
        map<int,vector<int>> subgraph; // store in form of adjacency list
        map<pair<int,int>,int> trusscity; // store initial trusscity value of each edge
        int edgeNumbers = 0;
        for(int vertex = 0;vertex < vertices;vertex++)
        {
            int vertexNumber =0,edgeCount = 0;
            MPI_Recv(&vertexNumber,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
            MPI_Recv(&edgeCount,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
            int neighbours[edgeCount];
            MPI_Recv(neighbours,edgeCount,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
            subgraph[vertexNumber] = vector<int>();
            for(int edge = 0;edge < edgeCount ;edge++)
            {
                subgraph[vertexNumber].push_back(neighbours[edge]);
                trusscity[make_pair(vertex,neighbours[edge])] = 2; // 2 will be the value initially
                edgeNumbers++;
            }
        }
        // graph distribution done and degrees also sent
        cout << "degree valu is : " << degrees[10] << endl;
        cout << "Vertices Received : "<< vertices << endl;
        cout << "Edges Received : " << edgeNumbers << endl;
        // now initialize trussities and start with the iterations
        vector<vector<int>> triangles (size-1,vector<int>()); // we store the triangles to be sent to each process (1 to size-1)
        for(auto& node : subgraph)
        {
            vector<int>& adjList = node.second;
            for(int i =0;i < adjList.size();i++)
            {
                for(int j =i+1;j < adjList.size();j++)
                {
                    if(degrees[adjList[i]]<degrees[adjList[j]] || (degrees[adjList[i]] == degrees[adjList[j]] && adjList[i] < adjList[j]))
                    {
                        // send the triangle to adjList[i]'s process which is adjList[i]%(size-1)+1
                        vector<int>& triangle = triangles[adjList[i]%(size-1)];
                        triangle.push_back(node.first);
                        triangle.push_back(adjList[i]);
                        triangle.push_back(adjList[j]); // 1 full triangle pushed (check i j)
                    }
                    else
                    {
                        vector<int>& triangle = triangles[adjList[j]%(size-1)];
                        triangle.push_back(node.first);
                        triangle.push_back(adjList[j]); // 1 full triangle pushed (chech j i)
                        triangle.push_back(adjList[i]);
                    }
                }
            }
        }
        // now we send these triangles count to coordinator process
        int countTriangleEnum [size-1] {};
        for(int i =0;i < size-1;i++)
        {
            countTriangleEnum[i] = triangles[i].size();
        }
        MPI_Send(countTriangleEnum,size-1,MPI_INT,0,rank,MPI_COMM_WORLD); // data sent
        int updationCount[size-1];
        MPI_Recv(updationCount,size-1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
        cout << "Receiving done!\n";
        for(int i =0;i< size-1;i++)
        {
            if(i+1 != rank)
            {
                MPI_Send(triangles[i].data(),countTriangleEnum[i],MPI_INT,i+1,i+1,MPI_COMM_WORLD);
            }
        }
        // sending to data done now receive first from the coordinator then update
        vector<vector<int>> updationTriangles;
        vector<vector<int>> replyTriangles(size-1,vector<int>()); // we keep track of updated triangles here and send them back 
        for(int i =1;i<size;i++)
        {
            if(i!= rank)
            {
                vector<int> process_i_triangle(updationCount[i-1]);
                MPI_Recv(process_i_triangle.data(),updationCount[i-1],MPI_INT,i,i,MPI_COMM_WORLD,&status);
                updationTriangles.push_back(process_i_triangle);
            }
        }
        // have'nt taken care of triangle updation in same process
    }
    MPI_Finalize();
    return 0;
}