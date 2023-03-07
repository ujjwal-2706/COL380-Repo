#include <mpi.h>
#include <iostream>
#include <fstream>
#include <bits/stdc++.h>
using namespace std;

string INPUT = "./TestCases/test4/test-input-4.gra";
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
        MPI_Status status;
        // now we will do the triangle enumeration loop by storing the count table from each process 1 to size-1
        for(int i=0;i<size-1;i++) // send to ith by collecting all from jth
        {
            vector<int> triangle_to_ith;
            for(int j =0;j < size-1;j++)
            {
                if(j != i)
                {
                    int triangle_count;
                    MPI_Recv(&triangle_count,1,MPI_INT,j+1,j+1,MPI_COMM_WORLD,&status);
                    vector<int> triangle_from_jth(triangle_count,-1);
                    MPI_Recv(triangle_from_jth.data(),triangle_count,MPI_INT,j+1,j+1,MPI_COMM_WORLD,&status);
                    triangle_to_ith.insert(triangle_to_ith.end(),triangle_from_jth.begin(),triangle_from_jth.end());
                }
            }
            int triangle_send = triangle_to_ith.size();
            MPI_Send(&triangle_send,1,MPI_INT,i+1,i+1,MPI_COMM_WORLD);
            MPI_Send(triangle_to_ith.data(),triangle_send,MPI_INT,i+1,i+1,MPI_COMM_WORLD);
        }

        for(int i=0;i<size-1;i++) // send to ith by collecting all from jth
        {
            vector<int> triangle_to_ith;
            for(int j =0;j < size-1;j++)
            {
                if(j != i)
                {
                    int triangle_count;
                    MPI_Recv(&triangle_count,1,MPI_INT,j+1,j+1,MPI_COMM_WORLD,&status);
                    vector<int> triangle_from_jth(triangle_count,-1);
                    MPI_Recv(triangle_from_jth.data(),triangle_count,MPI_INT,j+1,j+1,MPI_COMM_WORLD,&status);
                    triangle_to_ith.insert(triangle_to_ith.end(),triangle_from_jth.begin(),triangle_from_jth.end());
                }
            }
            int triangle_send = triangle_to_ith.size();
            MPI_Send(&triangle_send,1,MPI_INT,i+1,i+1,MPI_COMM_WORLD);
            MPI_Send(triangle_to_ith.data(),triangle_send,MPI_INT,i+1,i+1,MPI_COMM_WORLD);
        }
        
    }
    else // other processes
    {
        int n=0; // total vertices in graph
        int vertices = 0; // vertices received by process
        MPI_Status status;
        MPI_Recv(&n,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
        vector<int> degrees(n,-1);
        MPI_Recv(degrees.data(),n,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
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
        cout << "degree value is : " << degrees[10] << endl;
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
                        if(adjList[i]%(size-1)+1 == rank) // triangle in same process
                        {
                            if(trusscity.count(make_pair(adjList[i],adjList[j]))>0)
                            {
                                trusscity[make_pair(node.first,adjList[i])]++;
                                trusscity[make_pair(node.first,adjList[j])]++;
                                trusscity[make_pair(adjList[i],adjList[j])]++;
                            }
                        }
                        else
                        {
                            triangle.push_back(node.first);
                            triangle.push_back(adjList[i]);
                            triangle.push_back(adjList[j]); // 1 full triangle pushed (check i j)
                        }
                    }
                    else
                    {
                        vector<int>& triangle = triangles[adjList[j]%(size-1)];
                        if(adjList[j]%(size-1)+1 == rank) // triangle in same process
                        {
                            if(trusscity.count(make_pair(adjList[j],adjList[i]))>0)
                            {
                                trusscity[make_pair(node.first,adjList[i])]++;
                                trusscity[make_pair(node.first,adjList[j])]++;
                                trusscity[make_pair(adjList[j],adjList[i])]++;
                            }
                        }
                        else
                        {
                            triangle.push_back(node.first);
                            triangle.push_back(adjList[j]);
                            triangle.push_back(adjList[i]); // 1 full triangle pushed (check j i)
                        }
                    }
                }
            }
        }
        // now we send these triangles count to coordinator process
        vector<int> initial_triangle_updation;
        for(int j =0;j <size-1;j++)
        {
            if(j+1 != rank) // we send triangles
            {
                int triangle_count = triangles[j].size();
                MPI_Send(&triangle_count,1,MPI_INT,0,rank,MPI_COMM_WORLD);
                vector<int>& triangle = triangles[j];
                MPI_Send(triangle.data(),triangle_count,MPI_INT,0,rank,MPI_COMM_WORLD);
            }
            else // we receive triangles
            {
                int triangle_received = -1;
                MPI_Recv(&triangle_received,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
                vector<int> triangles_to_update(triangle_received,-1);
                MPI_Recv(triangles_to_update.data(),triangle_received,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
                initial_triangle_updation.insert(initial_triangle_updation.end(),triangles_to_update.begin(),triangles_to_update.end());
            }
        }
        triangles.clear(); // free up space
        cout << "Triangles received by :" << rank << " are "<< initial_triangle_updation.size() << endl;
        vector<vector<int>> segregate_updation(size-1,vector<int>()); // segregate which updation sent to which process
        for(int i =0;i <initial_triangle_updation.size();i+=3)
        {
            vector<int> update_triangle = {initial_triangle_updation[i],initial_triangle_updation[i+1],initial_triangle_updation[i+2]};
            if(trusscity.count(make_pair(update_triangle[1],update_triangle[2]))>0) // update
            {
                trusscity[make_pair(update_triangle[1],update_triangle[2])]++;
                update_triangle.push_back(1); // means true
            }
            else
            {
                update_triangle.push_back(0); // false
            }
            int process = initial_triangle_updation[0]%(size-1);
            segregate_updation[process].insert(segregate_updation[process].end(),update_triangle.begin(),update_triangle.end()); 
        }
        initial_triangle_updation.clear();
        for(int j =0;j <size-1;j++)
        {
            if(j+1 != rank) // we send triangles
            {
                int triangle_count = segregate_updation[j].size();
                MPI_Send(&triangle_count,1,MPI_INT,0,rank,MPI_COMM_WORLD);
                vector<int>& triangle = segregate_updation[j];
                MPI_Send(triangle.data(),triangle_count,MPI_INT,0,rank,MPI_COMM_WORLD);
            }
            else // we receive triangles
            {
                int triangle_received = -1;
                MPI_Recv(&triangle_received,1,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
                vector<int> triangles_to_update(triangle_received,-1);
                MPI_Recv(triangles_to_update.data(),triangle_received,MPI_INT,0,rank,MPI_COMM_WORLD,&status);
                initial_triangle_updation.insert(initial_triangle_updation.end(),triangles_to_update.begin(),triangles_to_update.end());
            }
        }
        // finally we update the triangles as received
        for(int i=0;i < initial_triangle_updation.size();i+=4)
        {
            vector<int> triangle = {initial_triangle_updation[i],initial_triangle_updation[i+1],initial_triangle_updation[i+2],initial_triangle_updation[i+3]};
            if(triangle[3] == 1) // do updation of uv and uw
            {
                trusscity[make_pair(triangle[0],triangle[1])]++;
                trusscity[make_pair(triangle[0],triangle[2])]++;
            }
        }
        initial_triangle_updation.clear();
        // triangle updation phase complete
        for(auto node : trusscity)
        {
            cout << "Trusscity of : " << node.first.first << " and "<< node.first.second << " is : " << trusscity[node.first] << endl;   
        }
        // now need to handle same edge counted multiple times in case both ends of same process
    }
    MPI_Finalize();
    return 0;
}