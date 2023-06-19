#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <time.h>
#include <vector>
#include <chrono>
using namespace std::chrono;
using namespace std;
int main()
{
    // vector<int> a = {20, 24, 24, 15, 25, 14, 14, 67, 54, 3, 7, 33, 7, 4, 59, 6, 77, 65, 23, 57, 135, 167, 34, 56, 443, 76, 89, 90, 23, 56};
    // vector<int> b = {15, 34, 31, 4, 56, 67, 35, 36, 36, 56, 89, 56, 78, 447, 788, 45, 66, 34, 79, 890, 44, 67, 24, 46, 568, 367, 456, 32, 57, 45};
    // vector<int> c(30);
    int N=10000000 ;
    // This program will create some sequence of random
    // numbers on every program run within range N
    vector<int> a;
    vector<int> b;
    vector<int> c(10000000);
    for (int i = 0; i < 10000000; i++)
        a.push_back(rand());
    for (int i = 0; i < 10000000; i++)
        b.push_back(rand());

    // Addition of two vectors
    auto start = high_resolution_clock::now();
    for (int i = 0; i < N; i++)
    {
        c[i] = b[i] + a[i];
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    // double time_taken=2;
    cout <<"Serial vector addition: "<< duration.count() << endl;
    start = high_resolution_clock::now();
#pragma omp parallel
    {
#pragma omp for
        {
            for (int i = 0; i < N; i++)
            {
                c[i] = b[i] + a[i];
            }
        }
    }
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    // double time_taken=2
    cout <<"Parallel vector addition: "<< duration.count()  << endl;

cout<<"***********************************"<<endl;
N=700;
// Multiply Vector and matrix
int A[N][N];
for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
        A[i][j]=rand();

    }
}
vector<int> d(N);
// serial
start = high_resolution_clock::now();
for(int i=0;i<N;i++){
    int sum=0;
    for(int j=0;j<N;j++){
        sum+=A[i][j]*a[j];
    }
    d[i]=sum;
}
stop = high_resolution_clock::now();
duration = duration_cast<microseconds>(stop - start);
cout<<"Serial Multiply Vector and matrix : "<< duration.count() << endl;

// parllel
start = high_resolution_clock::now();
#pragma omp parallel default(none)\
shared(A,a,N,c) private(i,j,sum) omp_set_num_threads(6)
for(int i=0;i<N;i++){
    int sum=0;
    for(int j=0;j<N;j++){
        sum+=A[i][j]*a[j];
    }
    d[i]=sum;
}
stop = high_resolution_clock::now();
duration = duration_cast<microseconds>(stop - start);
cout<<"Parallel Multiply Vector and matrix: "<< duration.count() << endl;

cout<<"*******************************"<<endl;

N=100;
A[N][N];
for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
        A[i][j]=rand();

    }
}
int B[N][N],C[N][N];
for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
        B[i][j]=rand();

    }
}
int i,j,k;
start = high_resolution_clock::now();
for( i=0;i<N;i++){
    for( j=0;j<N;j++){
        C[i][j]=0;
        for( k=0;k<N;k++){
            C[i][j]+=A[i][k]*B[k][j];
        }
        
    }
}
stop = high_resolution_clock::now();
duration = duration_cast<microseconds>(stop - start);
cout<<"Serial Multiply Matrix and matrix : "<< duration.count() << endl;

start = high_resolution_clock::now();
#pragma omp parallel default(private)\
shared(A,B,N,C) omp_set_num_threads(4)
#pragma omp schedule(static)
for( i=0;i<N;i++){
    for( j=0;j<N;j++){
        C[i][j]=0;
        for( k=0;k<N;k++){
            C[i][j]+=A[i][k]*B[k][j];
        }
        
    }
}
stop = high_resolution_clock::now();
duration = duration_cast<microseconds>(stop - start);
cout<<"Parallel Multiply Matrix and matrix : "<< duration.count() << endl;

    return 0;
    
}