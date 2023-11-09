#include<iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <vector>
#include <queue>
#include <cuda_runtime.h>
#include <cuda.h>
#include <ctime>
#include <chrono>
#include <random>
#include <cmath>

#define FALSE 0
#define TRUE 1
#define INF INT_MAX

bool verify(const std::vector<int>& arr1, const std::vector<int>& arr2)
{
  bool flag = 1;
  if(arr1.size() == arr2.size())
    std::cout<<"Step 1, i.e. verifying of size is complete : \n";
  else
    std::cout<<"Size unequal : ";
  for(int i = 0; i<arr1.size(); ++i)
  {
    if(arr1[i] != arr2[i])
    {
      std::cout<<"\nVALUE AT "<<i<<"IS DIFFERENT FOR BOTH THE ARRAYS\n";
      printf("Distance of parallel[%d] = %d whereas distance of serial[%d] = %d\n", i, arr1[i], i, arr2[i]);
      flag = 0;
      return flag;
    }

  }
  std::cout<<"\nCongratulatiions.. The bfs results are correct..\n";

  return flag;
}

std::vector<int> serialbfs(int src, const std::vector<std::vector<int>> &adjlist)
{
    int n = adjlist.size();
    std::vector<bool> visited_serial;
    std::queue<int>q;
    std::vector<int> dist_serial;

    visited_serial.resize(n);
    dist_serial.resize(n, INT_MAX);
    dist_serial[src] = 0;
    // int pos = 0;
    q.push(src);
    visited_serial[src] = 1;
    while(!q.empty())
    {
      int parent = q.front();
      q.pop();
      for(int i = 0; i<adjlist[parent].size(); i++)
      {
        if(visited_serial[adjlist[parent][i]] != 1)
        {
          q.push(adjlist[parent][i]);
          visited_serial[adjlist[parent][i]] = 1;
          
          if(dist_serial[adjlist[parent][i]] > dist_serial[parent] + 1)
            dist_serial[adjlist[parent][i]] = dist_serial[parent] + 1;
      }
    }
  }
  return dist_serial;
}

__global__ void BFS_KERNEL(int n, int *c_iteration_no, int *c_edgelist, int *c_csr_edge_range, int *c_dist, int *c_parent,int *c_visited, int* c_flag)
{
  // Your CUDA kernel implementation
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < n and c_visited[tid] == *c_iteration_no)
  {
    int vertex_no = tid;
    
    int start = c_csr_edge_range[vertex_no];
    int end = c_csr_edge_range[vertex_no+1];

    for(int j = start; j<end; ++j)
    {
      if(c_dist[c_edgelist[j]] > *c_iteration_no + 1)
      {
        int k = c_edgelist[j];
        c_dist[k] = *c_iteration_no+1;
        c_parent[k] = vertex_no;
        c_visited[k] = *c_iteration_no+1;
        *c_flag = 1;
      }
    }
  }
}

std::vector<int> BFS(const int s, const int n, const int e, const std::vector<std::vector<int>> &adjlist)
{
    std::vector<int> edgelist(2*e);
    std::vector<int> csr_edge_range(n+1); 
    std::vector<int> parent(n); 
  	std::vector<int> visited(n, -1);
  	std::vector<int> dist(n, INT_MAX);

    int flag = 1;
    int k = 0, iteration_no;

    //Build the CSR (Compact Sensitive Representation)
    csr_edge_range[0] = 0;
    for(int i = 0; i<adjlist.size(); ++i)
    {
      csr_edge_range[i+1] = csr_edge_range[i] + adjlist[i].size();
      for(int j = 0; j<adjlist[i].size(); ++j)
      {
        edgelist[k++] = adjlist[i][j];
      }
    }
    //Updation of Source distance and iteration number
    visited[s] = 0;
    dist[s] = 0;
    iteration_no = 0;

    //CUDA Variable initialization
    int *c_edgelist, *c_csr_edge_range, *c_visited, *c_dist, *c_parent, *c_flag,*c_iteration_no;

    auto new_start = std::chrono::high_resolution_clock::now();

    cudaMalloc((void**)&c_edgelist, sizeof(int)*(edgelist.size()));
    cudaMalloc((void**)&c_csr_edge_range, sizeof(int)*(csr_edge_range.size()));
    cudaMalloc((void**)&c_visited, sizeof(int) * n);
    cudaMalloc((void**)&c_dist, sizeof(int) * n);
    cudaMalloc((void**)&c_parent, sizeof(int) * n);
    cudaMalloc((void**)&c_flag, sizeof(int));
    cudaMalloc((void**)&c_iteration_no, sizeof(int));

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device); // get current device
    cudaGetDeviceProperties(&prop, device); // get the properties of the device

    int maxThreadsPerBlock = prop.maxThreadsPerBlock; // max threads that can be spawned per block

    // calculate the optimal number of threads and blocks
    int threadsPerBlock = (n < maxThreadsPerBlock) ? n : maxThreadsPerBlock;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaMemcpy(c_edgelist, edgelist.data(), sizeof(int)*(edgelist.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(c_csr_edge_range, csr_edge_range.data(), sizeof(int)*(csr_edge_range.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(c_visited, visited.data(), sizeof(int)*(visited.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(c_dist, dist.data(), sizeof(int)*(dist.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(c_parent, parent.data(), sizeof(int)*(parent.size()), cudaMemcpyHostToDevice);
    cudaMemcpy(c_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    while(flag) {
      flag = FALSE;
      cudaMemcpy(c_iteration_no, &iteration_no, sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(c_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);
      BFS_KERNEL<<<blocksPerGrid, threadsPerBlock>>> (n, c_iteration_no, c_edgelist, c_csr_edge_range, c_dist, c_parent, c_visited, c_flag);

      cudaMemcpy(&flag, c_flag, sizeof(int), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      iteration_no++;

      #ifdef DEBUG
        cout<<"iteration no and flag "<<iteration_no<<" "<<flag<<" ";
      #endif
    }
    cudaMemcpy(dist.data(), c_dist, sizeof(int)*(dist.size()), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Time for parallel bfs without copying the data: " << duration.count() << " milliseconds.\n";

    auto new_end = std::chrono::high_resolution_clock::now();
    auto new_duration = std::chrono::duration_cast<std::chrono::milliseconds>(new_end - new_start);

    std::cout << "Time for parallel bfs with copying the data: " << new_duration.count() << " milliseconds.\n";
    
    std::cout << "\nDepth of graph is : " << iteration_no + 1<<std::endl;
    std::cout << "log2"<<n <<" = "<<log2(n)<<std::endl;
    
    cudaMemcpy(parent.data(), c_parent, sizeof(int)*(parent.size()), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    parent[s] = s;

    int n_visited = 0;
    for(auto i : parent)
    {
      if(i != -1)
        n_visited++;
    }

    std::cout <<" The number of nodes that got visited are : "<< n_visited << std::endl;

    cudaFree(c_edgelist);
    cudaFree(c_csr_edge_range);
    cudaFree(c_visited);
    cudaFree(c_dist);
    cudaFree(c_parent);
    cudaFree(c_flag);
    cudaFree(c_iteration_no);

    return dist;
  }
int main(int argc, char** argv)
{
  std::random_device rd;
  std::mt19937 gen(rd());

  int src, n, e;
  src = std::atoi(argv[1]);
  n = std::atoi(argv[2]);
  e = std::atoi(argv[3]);

  std::vector<std::vector<int>> adjlist(n);
  int u,v;
  for(int i = 0; i<e; ++i)
  {
    std::uniform_int_distribution<int> distribution(0, n - 1);
    do {
      u = distribution(gen);
      v = distribution(gen);
    }
    while(u == v);
    adjlist[u].push_back(v);
    adjlist[v].push_back(u);
  }

  std::vector<int> dist = BFS(src,n,e,adjlist);

  auto new_start = std::chrono::high_resolution_clock::now();

  std::vector<int> serial_dist = serialbfs(src, adjlist);

  auto new_end = std::chrono::high_resolution_clock::now();
  auto new_duration = std::chrono::duration_cast<std::chrono::milliseconds>(new_end - new_start);

  std::cout << "Time for serial bfs : " << new_duration.count() << " milliseconds.\n";

  std::cout<<verify(dist, serial_dist);

  std::time_t currentTime = std::time(nullptr);
  char* timeString = std::ctime(&currentTime);

  // Print the current time
  std::cout << "Current time: " << timeString;


  return 0;
}