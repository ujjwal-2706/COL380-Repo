Number of Approaches: 4
1: Idea: In this approach, I multiplied the matrices by first creating a linear array of size n*n as well as storing the block value of each index i,j (whether the block is non zero or not) and then filled elements while reading the file and then multiplyed the whole matrix block by block by first checking if block is non zero or not then finally parallizing the outer for loop using tasks.
1: Why attempted: Since seeing for the possibility of false sharing is quite easy in case of arrays as well as the overhead of maintaining them is the least, that's why I used this approach.
1: Results: Speedup for matrix computation was around 1.5 times in case of small inputs.
1: Drawback: This approach was giving segmentation fault for large values of n since the sparse nature of the matrix wasn't fully exploited.

2: Idea: In this approach, in order to exploit the sparse nature of our matrix, I stored the matrix in the form of a vector which contains the map of Blocks of sparse matrix with j as the key and i as the vector index. Then we multiply the ith and jth rows of these vectors which contain the comman key k (hence exploiting symmetry) in order to compute the i,jth block of final answer. Finally we parallelize the outer for loop using tasks to obtain our final answer.
2: Why attempted: This approach was attempted in order to exploit the sparse nature of the matrix using which I was able to store large input sizes also in form of blocks without getting segmentation faults.
2: Results: Speedup for matrix computation was around 2 times in this case for large input sizes.
2: Drawback: In this approach, finding the common key k value for was having alot overhead since we have to check in the map of jth row each time for every map key of ith row.

3: Idea: In this approach, I stored our sparse matrix in the form of a nested map of maps where we get a i,j block by giving i as key for first map then j as key to map inside it, then we finally multiply it in a similar way as above apporach execept we don't take those i,j where either one of them is missing from the outer map. Finally we parallelize the map traversal using tasks.
3: Why attempted: Using this approach those unnecessary cases of i,j where any of i or j are empty block rows, will not contribute to computation and hence reduce time further.
3: Results: Speedup for matrix computation was around 2 times in this case for large input sizes.
3: Drawback: In this approach, we have to check the for both i and j to be present in maps (outer both) every time which in itself is not efficient compared to O(1) traversal.

4: Idea: In this final approach, I removed all the maps used and instead stored the sparse matrix as a vector of vector of blocks where we have i,j block stored in the ith row of the outer vector and inside it, it contains all the row blocks in sorted order of j. Then we multiply our matrix in order to obtain i,j block of final matrix by i, k and j,k row by row multiplication where we just have to check for common k value, which can be checked by two pointer technique while traversing the blocks of rows linearly ,finally we parallelize the outer for loop using tasks.
4: Why attempted: By using this approach, we don't need to check for k key value like we did in map every single time, thus increasing efficiency.
4: Results: Speedup for matrix computation was around 4 times in this case for large input sizes.
4: Drawback: In this approach, we are still taking those block by block matrix computation too in which only a few elements are non zero, thus this overhead can be optimized too.

Final scalability analysis: (On my local machine)
2^10, 2131, 3288.12, 2032.21, 1582.32, 1172.44
2^15, 426270, 22115.46, 14785.09, 13332.01, 13076.34
2^16, 860290, 19773.11, 13561.24, 9998.92, 9476.23
19993, 200024, 21878.12, 15344.22, 13417.47, 12385.46

Commentary: As we can see that upon increasing the number of cores the runtime for the whole program is decreasing at a good scale which reassures that the code has been efficiently parallelized to compute the matrix product. (Note that these runtimes are calculated using the time command in linux).
