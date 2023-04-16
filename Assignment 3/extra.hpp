#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>
using namespace std;


typedef unsigned long long ll;
ll t1 = pow(10 , 14);
ll t2 = pow(10 , 7);
ll t3 = pow(10 , 0);

ll p1 = pow(10 , 7);
ll p2 = pow(10 , 0);
int ni(ifstream& myFile);
void graph_input(string filename , int& n , int& m , vector<set<int>>& graph , vector<int>& degrees );
void graph_plus_creation(int n , int m , vector<set<int>>& graph , vector<int>& degrees , vector<vector<int>>& graph_plus , vector<int>& deg_plus);
int get_min_trusscity(unordered_map<ll,int>& current_trusscity,int size);
void exchange_triangles_update(unordered_map<ll,int>& current_trusscity,int current_min,
    unordered_map<ll,unordered_set<ll>>& supp_plus , unordered_map<ll,int>& final_trusscity , int vertices_per_process , int size);
string allInput(int argc,char *argv[]);
string inputParser(string& all_input,string input_to_find);

void update_connected_components(vector<int>& colors,unordered_map<int,vector<int>>& connected_components,vector<vector<int>>& sorted_trusscities,int& curr_pointer,int kval);
int ni(ifstream& myFile){
    unsigned char bytes[4];
    myFile.read((char *)&bytes , sizeof(bytes));
    return (bytes[0] | bytes[1]<<8 | bytes[2]<<16 | bytes[3]<<24);
}

void graph_input(string filename , int& n , int& m , vector<set<int>>& graph , vector<int>& degrees ){

    
    ifstream myFile(filename , ios::in | ios::binary);
    if(!myFile)printf("Incorrect File Name/Location\n");

    n=ni(myFile) , m = ni(myFile);

    degrees.resize(n);                             //we get the degree of ith node
    graph.resize(n,set<int>());
    // cout << n << " " << m << endl;
        
    for(int i = 0;i < n;i++)
    {
        int node = ni(myFile);int degree = ni(myFile);degrees[node] = degree;
        for(int edge = 0; edge< degree;edge++)graph[node].insert(ni(myFile)); 
    }
    myFile.close();
}

void graph_plus_creation(int n , int m , vector<set<int>>& graph , vector<int>& degrees , vector<vector<int>>& graph_plus , vector<int>& deg_plus){

    for(int i =0;i < n;i++)
    {
        for(auto &neighbour : graph[i])
        {
            if((degrees[i] < degrees[neighbour])|| (degrees[i] == degrees[neighbour] && i < neighbour))
            {
                graph_plus[i].push_back(neighbour);
            }
        }
        deg_plus[i] = graph_plus[i].size();
    }
}

int get_min_trusscity(unordered_map<ll,int>& current_trusscity,int size) // it will exchange info and get minimum iteration value
{
    int minimumTrusscity = INT_MAX;
    for(auto& edge : current_trusscity)
    {
        minimumTrusscity = min(minimumTrusscity,edge.second);
    }
    vector<int> send_buffer(size,minimumTrusscity);
    vector<int> recv_buffer(size,-1);
    MPI_Alltoall(send_buffer.data(),1,MPI_INT,recv_buffer.data(),1,MPI_INT,MPI_COMM_WORLD);
    int minValue= *min_element(recv_buffer.begin(),recv_buffer.end());
    return minValue; // If minimum returned is INT_MAX means all edges settled in all processes
}

void exchange_triangles_update(unordered_map<ll,int>& current_trusscity,int current_min,
    unordered_map<ll,set<vector<int>>>& supp_plus , unordered_map<ll,int>& final_trusscity , int vertices_per_process , int size)
{
    // filter out the triangles that need to be sent for deletion to correponding vertex
    vector<vector<int>> triangles_deletion(size,vector<int>()); // store the triangles of process i at index i in u,v,w format in a common vector
    vector<ll> store_edges;
    for(auto edge : current_trusscity)
    {
        if(edge.second == current_min) // delete all triangles corresponding to this edge
        {
            store_edges.push_back(edge.first);
            set<vector<int>>& triangle_of_edge = supp_plus[edge.first];
            for(auto& triangle : triangle_of_edge)
            {
                
                ll u = triangle[0] , v = triangle[1] , w = triangle[2];
                int process1 = min((int)u/vertices_per_process,size-1);
                int process2 = min((int)v/vertices_per_process,size-1);
                triangles_deletion[process1].insert(triangles_deletion[process1].end() , triangle.begin() , triangle.end());
                triangles_deletion[process2].insert(triangles_deletion[process2].end() , triangle.begin() , triangle.end());
            }
        }
    }
    for(auto edge : store_edges)
    {
        final_trusscity[edge]= current_min;
        current_trusscity.erase(edge);
    }
    store_edges.clear();
    int send_counts[size];
    int send_disp[size],recv_disp[size];
    int recv_counts[size];
    int offset = 0;
    for(int i =0;i < size;i++)
    {
        send_counts[i] = triangles_deletion[i].size();
        send_disp[i] = offset;
        offset += send_counts[i];
    }
    offset = 0;
    MPI_Alltoall(send_counts,1,MPI_INT,recv_counts,1,MPI_INT,MPI_COMM_WORLD);
    for(int i =0;i < size;i++)
    {
        recv_disp[i] = offset;
        offset += recv_counts[i];
    }
    vector<int> flattened_send;
    for(int i = 0;i < size;i++)
    {
        vector<int>& triangles = triangles_deletion[i];
        flattened_send.insert(flattened_send.end(),triangles.begin(),triangles.end());
    }
    triangles_deletion.clear(); // free up the space
    vector<int> flattened_recv(offset,-1);
    MPI_Alltoallv( flattened_send.data(),send_counts,send_disp, MPI_INT ,
                   flattened_recv.data(),recv_counts,recv_disp, MPI_INT , MPI_COMM_WORLD);

    flattened_send.clear();
    vector<int>().swap(flattened_send);
    // printf("Processing\n");
    // for(int i=0;i<flattened_recv.size();i++){cout<<flattened_recv[i]<<" ";}cout<<"\n";
    // We received the triangles as well as updated the trusscities
    for(int i = 0;i < flattened_recv.size();i += 3)
    {
        // printf("i = %d\n" , i);
        // ll triangle = flattened_recv[i];
        int u = flattened_recv[i] , v = flattened_recv[i+1] , w = flattened_recv[i+2];
        vector<int> triangle{u , v  , w};
        ll edge1 = p1*u+p2*v , edge2 = p1*u+p2*w , edge3 = p1*v+p2*w;
        if(supp_plus.count(edge1)>0 && supp_plus[edge1].count(triangle)>0)
        {
            // printf("Hello1\n");
            if(current_trusscity.count(edge1) > 0 && current_trusscity[edge1]>current_min) current_trusscity[edge1]--;
            supp_plus[edge1].erase(triangle);
        }
        if(supp_plus.count(edge2)>0 && supp_plus[edge2].count(triangle)>0)
        {
            // printf("Hello2\n");
            if(current_trusscity.count(edge2) > 0 && current_trusscity[edge2]>current_min) current_trusscity[edge2]--;
            supp_plus[edge2].erase(triangle);
        }
        if(supp_plus.count(edge3)>0 && supp_plus[edge3].count(triangle)>0)
        {
            // printf("Hello3\n");
            if(current_trusscity.count(edge3) > 0 && current_trusscity[edge3]>current_min) current_trusscity[edge3]--;
            supp_plus[edge3].erase(triangle);
        }
    }
    // printf("Processing Complete\n");
    // flattened_recv.clear(); // free up space
    vector<int>().swap(flattened_recv);
}

string allInput(int argc,char *argv[])
{
    string result = "";
    for(int i =1;i < argc;i++)
    {
        result.append(argv[i]);
    }
    return result;
}
string inputParser(string& all_input,string input_to_find)
{
    string answer;
    bool found = false;
    int index = all_input.find(input_to_find);
    if(index == -1) // substring not found
    {
        return ""; // empty string
    }
    index = index + input_to_find.size();
    int end_index = index;
    while(end_index < all_input.size())
    {
        if(end_index < all_input.size()-2 && all_input.substr(end_index+1,2) == "--")
        {
            break;
        }
        end_index++;
    }
    return all_input.substr(index,end_index-index+1);
}

void update_connected_components(vector<int>& colors,unordered_map<int,vector<int>>& connected_components,vector<vector<int>>& sorted_trusscities,int& curr_pointer,int kval)
{
    // keep on adding edges unless we find some edge whose trusscity lower then k + 2 or edges finish
    while(curr_pointer < sorted_trusscities.size())
    {
        vector<int> curr_edge = sorted_trusscities[curr_pointer];
        if(curr_edge[0] < kval + 2) break;
        else
        {
            int vertex1 = curr_edge[1], vertex2 = curr_edge[2];
            if(colors[vertex1] != colors[vertex2]) // merge the connected component
            {
                int del_color = colors[vertex2];
                int add_color = colors[vertex1];
                vector<int> add_vertices;
                for(int vertices : connected_components[del_color])
                {
                    colors[vertices] = add_color;
                    add_vertices.push_back(vertices);
                }
                vector<int>& add_vector = connected_components[add_color];
                add_vector.insert(add_vector.end(),add_vertices.begin(),add_vertices.end());
                add_vertices.clear();
                connected_components.erase(del_color);
            }
            curr_pointer++;
        }
    }
}

void update_parallel_dsu(int colors[],unordered_map<int,vector<int>>&  connected_components,vector<ll>& edges)
{
    for(ll edge : edges)
    {
        int vertex1 = (int) (edge/p1);
        int vertex2 = (int) (edge%p1);
        // join them together
        int color_add,color_del;
        // if((connected_components[colors[vertex1]]).size() > (connected_components[colors[vertex2]]).size())
        // {
        //     color_add = colors[vertex1];
        //     color_del = colors[vertex2];
        // }
        // else
        // {
        //     color_add = colors[vertex2];
        //     color_del = colors[vertex1];
        // }
        color_add = colors[vertex2];
        color_del = colors[vertex1];
        if(color_add != color_del)
        {
            vector<int> add_vertices;
            for(int vertices : connected_components[color_del])
            {
                colors[vertices] = color_add;
                add_vertices.push_back(vertices);
            }
            vector<int>& add_vector = connected_components[color_add];
            add_vector.insert(add_vector.end(),add_vertices.begin(),add_vertices.end());
            add_vertices.clear();
            connected_components.erase(color_del);
        }
    }
}

void parallel_dsu(int colors[],unordered_map<int,vector<int>>& connected_components,int rank,
    int size,int kval,unordered_map<ll,int>& final_trusscity)
{
    vector<ll> edge_to_send;
    for(auto& edge : final_trusscity)
    {
        if(edge.second >= kval + 2) // we need to send this edge
        {
            edge_to_send.push_back(edge.first);
        }
    }
    //cout << edge_to_send.size() << " edge size is : " << rank << endl;
    // we first select the edges to be sent then receive edges then update connected_components
    for(int i =0;i < size;i++) // node i is the sender of edges, others will receive and update
    {
        if(i == rank) // rank will send
        {
            int count=edge_to_send.size();
            //cout << "Count sent by rank : " << rank << " is : " << count << endl;
            MPI_Bcast(&count,1,MPI_INT,i,MPI_COMM_WORLD);
            MPI_Bcast(edge_to_send.data(),count,MPI_UNSIGNED_LONG_LONG,i,MPI_COMM_WORLD);
            update_parallel_dsu(colors,connected_components,edge_to_send);
        }
        else
        {
            int recv = 0;
            MPI_Bcast(&recv,1,MPI_INT,i,MPI_COMM_WORLD);
            //cout << "Count received by rank : " << rank << " is : " << recv << endl;
            vector<ll> recv_edge(recv);
            MPI_Bcast(recv_edge.data(),recv,MPI_UNSIGNED_LONG_LONG,i,MPI_COMM_WORLD);
            update_parallel_dsu(colors,connected_components,recv_edge);
        }
    }
}

int component_info_exchange(int comp_number,int size)
{
    // exchange and return true if any component
    int recv_buffer[size];
    int send_buffer[size];
    for(int i = 0;i < size;i++) send_buffer[i] = comp_number;
    int answer = 0;
    // cout << "Allto all start\n";
    MPI_Alltoall(send_buffer,1,MPI_INT,recv_buffer,1,MPI_INT,MPI_COMM_WORLD);
    // cout << "Component evaluated\n";
    for(int i =0;i < size;i++)
    {
        answer += recv_buffer[i];
    }
    return answer;
}

void graph_variable_read(string INPUT , int&n , int& m , int& graph_in_bytes){

    const char *graphname = INPUT.c_str();
    MPI_File graphfile;
    int graphmode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    int open_err = MPI_File_open(MPI_COMM_WORLD , graphname , graphmode , MPI_INFO_NULL , &graphfile);
    if(open_err != 0)cout<<"CANT OPEN GRAPH FILE\n";

    MPI_Offset offset;
    MPI_File_get_size(graphfile , &offset);
    graph_in_bytes = offset;

    int inputs[2] = {0};
    MPI_File_read_all(graphfile , inputs , 2 , MPI_INT , MPI_STATUS_IGNORE);
    MPI_File_close(&graphfile); 
    n = inputs[0] , m = inputs[1];
    return;
}

// READ Offsets for ranks from Header file
void graph_header_read(string HEADER , int n , vector<int>& offsets){
    
    const char *headername = HEADER.c_str();
    MPI_File headerfile;
    int headermode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    int _open_err = MPI_File_open(MPI_COMM_WORLD , headername , headermode , MPI_INFO_NULL , &headerfile);
    if(_open_err != 0)cout<<"CANT OPEN HEADER FILE\n";
    MPI_File_read_all(headerfile , offsets.data() , n , MPI_INT , MPI_STATUS_IGNORE);     
    MPI_File_close(&headerfile);
}

void graph_parallel_read(string INPUT , int rank , int rank_count , vector<int>& flattened_local_graph){

    const char *graphname = INPUT.c_str();
    MPI_File graphfile;
    int graphmode = MPI_MODE_CREATE | MPI_MODE_RDWR;
    int open_err = MPI_File_open(MPI_COMM_WORLD , graphname , graphmode , MPI_INFO_NULL , &graphfile);
    if(open_err != 0)cout<<"CANT OPEN GRAPH FILE\n";

    /* MOVE COMMON POINTER TO 8 BYTES */
    MPI_File_seek_shared(graphfile , 8 , MPI_SEEK_SET);

    int read_err = MPI_File_read_ordered(graphfile , flattened_local_graph.data() , rank_count , MPI_INT , MPI_STATUS_IGNORE);
    if(read_err !=0 )cout<<"CANT READ FILE\n";

    MPI_File_close(&graphfile); 
    return;
}

/* Create Local Graph From Flattened Local Graph */
void graph_create_local_graph_plus(vector<int>& flattened_local_graph , map<int , vector<int>>& local_graph 
    , vector<int>& local_deg,map<int,vector<int>>& adjacency_list){

    int flattened_graph_size = flattened_local_graph.size();
    int itr = 0;
    while(itr < flattened_graph_size){
        
        int u = flattened_local_graph[itr];
        int deg_u = flattened_local_graph[itr+1];

        local_graph[u] = vector<int>();
        adjacency_list[u] = vector<int>();
        for(int i = itr+2; i < itr+deg_u+2;i++){

            int neighbour = flattened_local_graph[i];
            if( ( local_deg[neighbour] > local_deg[u] ) || ( local_deg[neighbour]==local_deg[u] && u < neighbour) ){
                local_graph[u].push_back(neighbour);
            }
            adjacency_list[u].push_back(neighbour);
        }
        itr += deg_u+2;
    }

    return;   
}