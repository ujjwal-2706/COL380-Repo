#include <chrono>
#include "extra.hpp"

// Design
// 1. Input
// 2. Graph Creation 
// 3. Triangle Computation
// 4. Iterative t_cap(e) computation
// 5. k-group computation
// 6. Output


// Issues
// 1. NOTE : Keep everything as vector only.(Assigning big-arrays gives seg-fault as freed memory from previous vectors is not assigned to it somehow) 

int main(int argc , char* argv[]){
    string all_input = allInput(argc,argv);
    string INPUT = inputParser(all_input,"--inputpath=");
    string OUTPUT = inputParser(all_input,"--outputpath=");
    string HEADER = inputParser(all_input, "--headerpath=");
    int kmin;
    string temp_kmin = inputParser(all_input,"--startk=");
    int kmax = stoi(inputParser(all_input,"--endk="));
    kmin = (temp_kmin.size()!=0)?stoi(temp_kmin):kmax;
    string temp = inputParser(all_input,"--verbose=");
    int verbose = (temp.size() == 0)? 0 : stoi(temp);
    string temp2 = inputParser(all_input,"--taskid=");
    int taskid = temp2.size()==0? 1: stoi(temp2);
    int p;
    if(taskid == 2) p = stoi(inputParser(all_input,"--p="));
    int rank, size;                         /* rank = rank of process in group , size = size of communicator group */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* TIMED BEGIN*/
    auto begin = std::chrono::high_resolution_clock::now();
    auto graph_begin = std::chrono::high_resolution_clock::now();
    int n , m , graph_in_bytes , vertices_per_process;

    /* READ n , m , graph_in_bytes */
    graph_variable_read(INPUT , n , m , graph_in_bytes);
    vertices_per_process = n/size;
    // printf("RANK = %d , n = %d , m = %d , graph_in_bytes = %d \n" , rank , n , m , graph_in_bytes);

    /* READ HEADER FILE */
    /* Useful to extract degrees of vertices (else would have to do )*/
    /* Every rank should be alloted bytes between [offset1 , offset2-1]*/
    vector<int> offsets(n , 0);
    graph_header_read(HEADER , n , offsets);

    int offset1 , offset2;
    offset1 =  offsets[rank*vertices_per_process];
    offset2 =  offsets[(rank+1)*vertices_per_process];
    if(rank==size-1)offset2=graph_in_bytes;
    //printf("RANK = %d , offset1 = %d , offset2 = %d \n" , rank , offset1 , offset2);

    /* Setup Degrees */
    vector<int> local_deg(n , 0);
    for(int i=0;i<n-1;i++){
        local_deg[i] = (offsets[i+1]-offsets[i])/4-2;
    }
    local_deg[n-1] = (graph_in_bytes-offsets[n-1])/4-2;

    /* READ GRAPH FILE */
    int rank_count = (offset2-offset1)/4;
    vector<int> flattened_local_graph(rank_count);
    graph_parallel_read(INPUT , rank , rank_count , flattened_local_graph);
    
    /* CREATE LOCAL GRAPH_PLUS */
    map<int , vector<int>> local_graph_plus;
    map<int,vector<int>> adjacency_list;
    graph_create_local_graph_plus(flattened_local_graph , local_graph_plus , local_deg,adjacency_list);
    // flattened_local_graph.clear();     /* FREE-UP MEMORY */
    vector<int>().swap(flattened_local_graph);
    auto graph_end = std::chrono::high_resolution_clock::now();
    // printf("TIMED GRAPH = %lf secs , RANK = %d\n" , 1e-9 * (std::chrono::duration_cast<std::chrono::nanoseconds>(graph_end - graph_begin)).count() , rank);
    /* TIMED GRAPH */

    /**************************** TRIANGLE COMPUTATION *********************************/

    /* TIMED PRE-PROCESSING */
    auto preprocess_begin = std::chrono::high_resolution_clock::now();

    /* PREPROCESSING - which triangles to send to which process */
    
    /* triangles_to_processes[i] = triangles to be sent to ith processes from current process */
    unordered_map<int , vector<int> > triangles_to_processes;            
    for(auto &p : local_graph_plus){
        int u = p.first;
        for(int &v : p.second){
            int owner_v = min(v / vertices_per_process , size-1);
            for(int &w : p.second){
                if(local_deg[v] > local_deg[w] || (local_deg[v]==local_deg[w] && v>=w) ) continue;
                
                /* CASE : u < v < w */
                // ll triangle = t1*u + t2*v + t3*w;
                triangles_to_processes[owner_v].push_back(u);
                triangles_to_processes[owner_v].push_back(v);
                triangles_to_processes[owner_v].push_back(w);

            }
        }
    }

    auto preprocess_end = std::chrono::high_resolution_clock::now();
    // printf("TIMED PREPROCESS = %lf secs , RANK = %d\n" , 1e-9 * (std::chrono::duration_cast<std::chrono::nanoseconds>(preprocess_end - preprocess_begin)).count() , rank);
    /* TIMED PRE-PROCESSING */

    /* TIMED QUERY */
    auto query_begin = std::chrono::high_resolution_clock::now();

    /* QUERY_SEND_BUF SETUP */
    vector<int> query_send_buf;
    int query_send_counts[size] = {0};
    int query_send_dispels[size] = {0};
    int query_send_buf_size = 0;
    
    for(int i=0;i<size;i++){
    
        int owner_process = i ;
        query_send_dispels[owner_process] = query_send_buf_size;
        
        if(triangles_to_processes.count(owner_process)>0){
           
            query_send_buf.insert(query_send_buf.end() , triangles_to_processes[owner_process].begin() , triangles_to_processes[owner_process].end());
            query_send_buf_size += triangles_to_processes[owner_process].size();
        }
        query_send_counts[owner_process] = query_send_buf_size - query_send_dispels[owner_process];
    }

    unordered_map<int,vector<int>>().swap(triangles_to_processes);
    // triangles_to_processes.clear();              /* FREE UP MEMORY */
    
    /* INFO_EXCHANGE i.e how much data is gonna be sent to each process */

    int query_recv_counts[size] = {0};
    int query_recv_dispels[size] = {0};
    
    MPI_Alltoall(query_send_counts , 1 , MPI_INT , query_recv_counts , 1 , MPI_INT , MPI_COMM_WORLD);
    
    /* QUERY_RECV_BUF SETUP */ 
    int query_recv_buf_size = 0;
    for(int i = 0;i< size;i++){
        query_recv_dispels[i] = query_recv_buf_size;
        query_recv_buf_size += query_recv_counts[i];
    }

    // //printf("RANK %d , QUERY_RECV_BUF_SIZE = %d , QUERY_SEND_BUF_SIZE = %d\n" , rank , query_recv_buf_size , query_send_buf_size);
    vector<int> query_recv_buf(query_recv_buf_size);

    MPI_Alltoallv(query_send_buf.data() , query_send_counts , query_send_dispels , MPI_INT , 
                  query_recv_buf.data() , query_recv_counts , query_recv_dispels , MPI_INT , MPI_COMM_WORLD);
    
    vector<int>().swap(query_send_buf);
    // query_send_buf.clear();             /* FREE-UP MEMORY */
    
    auto query_end = std::chrono::high_resolution_clock::now();
    // printf("TIMED QUERY = %lf secs , RANK = %d\n" , 1e-9 * (std::chrono::duration_cast<std::chrono::nanoseconds>(query_end - query_begin)).count() , rank);
    /* TIMED QUERY */

    /* TIMED RESPONSE */
    auto response_begin = std::chrono::high_resolution_clock::now();

    /* PROCESSING QUERIES / RESPONSE_SEND_BUF SETUP */

    unordered_map<ll , set<vector<int>>> supp_plus;            /* supp_plus[e] = set of triangles incident on e */
    for(auto &p:local_graph_plus){
        int u = p.first;
        for(auto &v : p.second){
            supp_plus[p1*u +p2*v] = set<vector<int>>();
        }
    }

    map<int,vector<int>>().swap(local_graph_plus);
    vector<int>().swap(local_deg);
    // local_graph_plus.clear();local_deg.clear();  /* FREE-UP MEMORY */
    /* TIMED RESPONSE SETUP*/
    auto response_setup_begin = std::chrono::high_resolution_clock::now();


    /* NOTE : memory optimising by reusing query_recv_buffer to send response to queries as follows */
    /* NOTE : give response by setting values to (-1 , -1 , -1) if triangle not existent */

    
    for(int j = 0 ; j < query_recv_buf_size; j +=3){

        // ll triangle = query_recv_buf[j];
        int u = query_recv_buf[j] , v = query_recv_buf[j+1] , w = query_recv_buf[j+2];

        /* Triangle = {u , v , w} exists */
        if(supp_plus.count(p1*v + p2*w) > 0){
            supp_plus[p1*v + p2*w].insert(vector<int>{u , v , w});
        }
        else{
            query_recv_buf[j] = 0 , query_recv_buf[j+1] = 0 , query_recv_buf[j+2] = 0;
        }
    }
    

    auto response_setup_end = std::chrono::high_resolution_clock::now();
    // printf("TIMED RESPONSE SETUP = %lf secs , RANK = %d\n" , 1e-9 * (std::chrono::duration_cast<std::chrono::nanoseconds>(response_setup_end - response_setup_begin)).count() , rank);
    /* TIMED RESPONSE SETUP*/

    /* RESPONSE_RECV_BUF SETUP */

    vector<int> response_recv_buf(query_send_buf_size);
    int response_recv_counts[size];copy(query_send_counts , query_send_counts+size , response_recv_counts);
    int response_recv_dispels[size];copy(query_send_dispels , query_send_dispels+size ,  response_recv_dispels);

    MPI_Alltoallv(query_recv_buf.data() , query_recv_counts , query_recv_dispels , MPI_INT , 
                  response_recv_buf.data() , response_recv_counts , response_recv_dispels , MPI_INT , MPI_COMM_WORLD);

    vector<int>().swap(query_recv_buf);
    // query_recv_buf.clear();            /* FREE-UP MEMORY */

    auto response_end = std::chrono::high_resolution_clock::now();
    // printf("TIMED RESPONSE = %lf secs , RANK = %d\n" , 1e-9 * (std::chrono::duration_cast<std::chrono::nanoseconds>(response_end - response_begin)).count() , rank);
    /* TIMED RESPONSE */

    /* PROCESS RESPONSE , UPDATE TRIANGLES IN supp_plus */

    /* TIMED RESPONSE PROCESS */
    auto response_process_begin = std::chrono::high_resolution_clock::now();

    for(int i=0;i<query_send_buf_size;i+=3){

        if(response_recv_buf[i]==0)continue;
 
        // ll triangle = response_recv_buf[i];
        int u = response_recv_buf[i] , v = response_recv_buf[i+1] , w = response_recv_buf[i+2];

        supp_plus[p1*u + p2*v].insert(vector<int>{u , v , w});
        supp_plus[p1*u + p2*w].insert(vector<int>{u , v , w});
        
    }
    vector<int>().swap(response_recv_buf);
    // response_recv_buf.clear();       /* FREE-UP MEMORY */

    auto response_process_end = std::chrono::high_resolution_clock::now();
    // printf("TIMED RESPONSE PROCESS = %lf secs , RANK = %d\n" , 1e-9 * (std::chrono::duration_cast<std::chrono::nanoseconds>(response_process_end - response_process_begin)).count() , rank);
    /* TIMED RESPONSE PROCESS */

    /**************************** TRIANGLE COMPUTATION *********************************/
    
    /* TIMED ITERATION */
    auto iteration_begin = std::chrono::high_resolution_clock::now();
    
    /*Now we will start with the iteration */
    unordered_map<ll,int> current_trusscity; // To be used during iteration
    unordered_map<ll,int> final_trusscity;
    for(auto& edges: supp_plus)
    {
        current_trusscity[edges.first] = (edges.second).size()+2; // set Trusscity to suppE + 2 
        // printf("Edge [%lld , %lld] , Trussity = %ld\n" , edges.first/p1 , edges.first%p1 , edges.second.size()+2);
    }
    int iter = 0;
    while(true)
    {
        
        /* TIMED ITERATION */
        auto iteration_per_begin = std::chrono::high_resolution_clock::now();

        int current_min = get_min_trusscity(current_trusscity,size);
        if(current_min > kmax+2) break;
        exchange_triangles_update(current_trusscity,current_min,supp_plus,final_trusscity,vertices_per_process,size);
    
        auto iteration_per_end = std::chrono::high_resolution_clock::now();iter++;
        // printf("K = %d , RANK = %d , ITERATION = %d , TIMED PER - ITERATION = %lf secs\n" , current_min , rank , iter , 1e-9 * (std::chrono::duration_cast<std::chrono::nanoseconds>(iteration_per_end - iteration_per_begin)).count());
        /* TIMED ITERATION */

    }
    unordered_map<ll,set<vector<int>>>().swap(supp_plus);
    // supp_plus.clear();
    for(auto edge : current_trusscity)
    {
        final_trusscity[edge.first] = edge.second;
    }
    unordered_map<ll,int>().swap(current_trusscity);
    // current_trusscity.clear();
    
    auto iteration_end = std::chrono::high_resolution_clock::now();
    // printf("TIMED ITERATION = %lf secs , RANK = %d\n" , 1e-9 * (std::chrono::duration_cast<std::chrono::nanoseconds>(iteration_end - begin)).count() , rank);
    // Zero's Code Start for parallel DSU
    //cout << "N value is : " << n << endl;
    //cout << final_trusscity.size() << " " << rank << endl;
    MPI_File output_file;
    MPI_File_open(MPI_COMM_WORLD,OUTPUT.c_str(),MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&output_file);
    if(taskid== 1)
    {
        for(int kval = kmin;kval <= kmax;kval++)
        {
            int* colors = new int[n];
            unordered_map<int,vector<int>> connected_components;
            for(int i =0; i < n;i++)
            {
                colors[i] = i;
                connected_components[i] = vector<int> {i};
            }
            parallel_dsu(colors,connected_components,rank,size,kval,final_trusscity);
            // now rank i will traverse the color value corresponding to its vertices
            delete(colors);
            vector<int> connected_color_number;
            if(rank != size-1)
            {
                for(int vertex= vertices_per_process*rank;vertex <vertices_per_process*(rank+1);vertex++)
                {
                    if(connected_components.count(vertex)>0 && (connected_components[vertex]).size()>1)
                    connected_color_number.push_back(vertex);
                }
            }
            else
            {
                for(int vertex = vertices_per_process*rank;vertex < n;vertex++)
                {
                    if(connected_components.count(vertex)>0 && (connected_components[vertex]).size()>1)
                    connected_color_number.push_back(vertex);
                }
            }
            int total_comp = 0;
            for(int i =0;i < n;i++)
            {
                if(connected_components.count(i)>0 && (connected_components[i]).size() > 1) total_comp++;
            }
            //cout << "Total Comp is : " << total_comp << " rank: " << rank<< endl;
            int ifAny = component_info_exchange(connected_color_number.size(),size);
            string write_buffer = "";
            if(rank == 0)
            {
                if(ifAny>0 && verbose == 1)
                {
                    write_buffer.append("1\n");
                    write_buffer.append(to_string(ifAny));
                    write_buffer.push_back('\n');
                }
                else if(ifAny > 0 && verbose == 0)
                {
                    write_buffer.append("1 ");
                }
                else if(ifAny == 0 && verbose == 0)
                {
                    write_buffer.append("0 ");
                }
                else
                {
                    write_buffer.append("0\n");
                }
            }
            if(verbose ==1)
            {
                for(int color_val : connected_color_number)
                {
                    vector<int> component = connected_components[color_val];
                    for(int val : component)
                    {
                        write_buffer.append(to_string(val));
                        write_buffer.push_back(' ');
                    }
                    write_buffer.push_back('\n');
                }
            }
            // //cout << "I am writing! " << rank << " " << write_buffer.size()<< endl; 
            //cout << ifAny << " "<< rank <<  endl;
            MPI_File_write_ordered(output_file,write_buffer.c_str(),write_buffer.size(),MPI_CHAR,MPI_STATUS_IGNORE);
        }
    }
    else if(taskid==2)
    {
        int* colors = new int[n];
        unordered_map<int,vector<int>> connected_components;
        for(int i =0; i < n;i++)
        {
            colors[i] = i;
            connected_components[i] = vector<int> {i};
        }
        parallel_dsu(colors,connected_components,rank,size,kmax,final_trusscity);
        // now rank i will traverse the vertex value corresponding to its vertices and check if influencer
        int end_vertex = vertices_per_process*(rank+1);
        if(rank == size-1) end_vertex = n;
        vector<int> influencers;
        for(int vertex = vertices_per_process*rank;vertex < end_vertex;vertex++)
        {
            // check if influencer or not
            unordered_set<int> color_present;
            for(int neighbour : adjacency_list[vertex])
            {
                if((connected_components[colors[neighbour]]).size() > 1)
                    color_present.insert(colors[neighbour]);
                if(color_present.size() >= p)
                {
                    influencers.push_back(vertex);
                    break;
                }
            }
        }
        int total_influencers = component_info_exchange(influencers.size(),size);
        string write_buffer = "";
        if(rank == 0)
        {
            write_buffer.append(to_string(total_influencers));write_buffer.push_back('\n');
        }
        if(verbose == 1)
        {
            for(int vertex: influencers)
            {
                unordered_set<int> color_done;
                write_buffer.append(to_string(vertex));write_buffer.push_back('\n');
                for(int neighbour : adjacency_list[vertex])
                {
                    if(color_done.count(colors[neighbour])== 0 && (connected_components[colors[neighbour]]).size() > 1)
                    {
                        for(int num : connected_components[colors[neighbour]])
                        {
                            write_buffer.append(to_string(num));write_buffer.push_back(' ');
                        }
                        color_done.insert(colors[neighbour]);
                    }
                }
                write_buffer.push_back('\n');
            }
        }
        else
        {
            for(int vertex: influencers)
            {
                write_buffer.append(to_string(vertex));write_buffer.push_back(' ');
            }
        }
        delete(colors);
        MPI_File_write_ordered(output_file,write_buffer.c_str(),write_buffer.size(),MPI_CHAR,MPI_STATUS_IGNORE);
    }
    // parallel DSU end -----------------------------------------------------------------------------
    /* TIMED ITERATION */

    /* OUTPUT SEND TO PROCESS SIZE - 1 */
    /* Strategy  : Send edge trussity(edge) to each process */

    /*vector<ll> truss_send_buf;
    int truss_send_counts[size] = {0};
    int truss_send_dispels[size] = {0};

    vector<ll> truss_recv_buf;
    int truss_recv_counts[size] = {0};
    int truss_recv_dispels[size] = {0};

    for(auto &p : final_trusscity){ 
        truss_send_buf.push_back(p.first);
        truss_send_buf.push_back(p.second);
    }    

    truss_send_counts[size-1] = truss_send_buf.size();

    MPI_Alltoall(truss_send_counts , 1 , MPI_INT , 
                 truss_recv_counts , 1 , MPI_INT , MPI_COMM_WORLD);
    
    int truss_recv_buf_size = 0;
    for(int i=0;i<size;i++){
        truss_recv_dispels[i] = truss_recv_buf_size;
        truss_recv_buf_size += truss_recv_counts[i]; 
    }
    truss_recv_buf.resize(truss_recv_buf_size);

    MPI_Alltoallv(truss_send_buf.data() , truss_send_counts , truss_send_dispels , MPI_UNSIGNED_LONG_LONG , 
                  truss_recv_buf.data() , truss_recv_counts , truss_recv_dispels , MPI_UNSIGNED_LONG_LONG , MPI_COMM_WORLD );

    truss_send_buf.clear();

    if(rank==size-1){

        for(int i=0;i<truss_recv_buf_size;i+=2){
            ll edge = truss_recv_buf[i] , trussity = truss_recv_buf[i+1];
            final_trusscity[edge] = trussity;
        }

        vector<int> degrees;vector<set<int>> graph;
        graph_input(INPUT , n , m , graph , degrees);

        if(taskid==1)
        {
            ofstream outFile(OUTPUT);
            for(int i=kmin;i<=kmax;i++){
                dsu_ktruss(n , m , i+2 , graph , final_trusscity , verbose , outFile);
            }
            outFile.close();
        }
        else // need to compute the influencers for each k value by creating connected components first and iterating over vertices
        {
            // sort the edges first in decreasing order of truss values
            // for a given k mark only those edges with truss value greater then equal to k+2
            // apply dsu  to find colors of new connected components, once connected components found, find influencers
            // next iteration, decrease k and add new edges and update connected components then again find influencers
            vector<int> colors(n,-1); for(int i =0;i < n;i++) colors[i] = i;
            vector<vector<int>> sorted_trusscities; for(auto& edge: final_trusscity) sorted_trusscities.push_back(vector<int>{edge.second,(int) (edge.first/p1),(int) (edge.first%p1)});
            sort(sorted_trusscities.begin(),sorted_trusscities.end(),greater<vector<int>>()); // decreasing order sort
            unordered_map<int,vector<int>> connected_components; for(int i =0;i < n;i++) connected_components[i] =vector<int>{i};
            int curr_pointer = 0;
            for(int kval = kmax;kval >= kmin;kval--)
            {
                update_connected_components(colors,connected_components,sorted_trusscities,curr_pointer,kval);
                // now we will find influencers
                vector<int> influencers;
                for(int vertex = 0;vertex < n;vertex++)
                {
                    set<int>& neighbours = graph[vertex];
                    set<int> color_observed;
                    for(int neighbour : neighbours)
                    {
                        if(connected_components[colors[neighbour]].size() > 1) color_observed.insert(colors[neighbour]);
                        if(color_observed.size() == p) break;
                    }
                    if(color_observed.size() >= p) influencers.push_back(vertex);
                }
                //cout << "Influencers are : " << influencers.size() << endl;
                for(int val : influencers) //cout << val << endl;
            }

        }
    }
    */
    /* TIMED END */
    MPI_File_close(&output_file);
    auto end = std::chrono::high_resolution_clock::now();
    //printf("TIMED WORK = %lf secs , RANK = %d\n" , 1e-9 * (std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)).count() , rank);
    /* TIMED END */

    MPI_Finalize();
    return 0;
}
