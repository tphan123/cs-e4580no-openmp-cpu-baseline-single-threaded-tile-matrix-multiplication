cs-e4580 no-openmp-cpu-baseline-multithreaded-tile-matrix-multiplication

[v0.4.0] - commit 29fefd0

i8mm.cc: included strip mining for row tile element ij, column tile leemnt kj, and inner tile element jj. 
i8mm.cc: passed all unit matrix tests and benchmarks at --no-timeout.

```
#include "mm.h"

void gemm ( int m , int n , int k ,
const std :: int8_t A , const std :: int8_t B , std :: int32_t C ) {
    for (int i = 0; i < m ; ++ i) {
        for (int ij = 0, ij < jc, ++ij){

         
            for (int kk = 0; kk < k ; ++ kk ) {
                for (int kj = 0, kj < kc, ++ij){


                    
                    for (int j = 0; j < n ; ++ j ) {
                        for(int jj = 0, jj < jc, ++jj ){
                            C[i*n+j] += A[i*k+kk]*B[kk*n+j];

                }
            } 
                }
            }
        }
    } 
}
```

 [v0.3.0] - commit 822fc49
i8mm.cc: added int tilesize, added int kk_tileend to account for partial tiles, added acc to account for accumulation of the sum product to meet no zero matrix C initiation requirement of the test, initiate matrix C only at the beginning as the first iterated accumulation. int32_t acc = 0 added. i8mm.cc: passed all unit matrix tests and benchmarks at --no-timeout.
```
#include "mm.h"

void gemm ( int m , int n , int k ,
const std :: int8_t A , const std :: int8_t B , std :: int32_t C ) {
    for (int i = 0; i < m ; ++ i) {
        for (int ij = 0, ij < jc, ++ij){

         
            for (int kk = 0; kk < k ; ++ kk ) {
                for (int kj = 0, kj < kc, ++ij){


                    
                    for (int j = 0; j < n ; ++ j ) {
                        for(int jj = 0, jj < jc, ++jj ){
                            C[i*n+j] += A[i*k+kk]*B[kk*n+j];

                }
            } 
                }
            }
        }
    } 
}
```


[v0.2.0] loop interchanging to avoid cache miss from skipping from one row and one tile element and one column at a time. 
solved it by iterating one row at pivot i, through the entire elements of the row at pivot k before iterating through the next column with pivot j. 
```
for (int i = 0; i < m; ++i) {
    for (int kk = 0; kk < k; ++kk) {
        for (int j = 0; j < n; ++j) {
            C[i*n + j] += A[i*k + kk] * B[kk*n + j];
        }
    }
}
```


[v0.1.0] - naive implementation nested loop 
```
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
        std::int32_t sum = 0;
        for (int kk = 0; kk < k; ++kk) {
            sum += A[i*k + kk] * B[kk*n + j];
        }
        C[i*n + j] = sum;
    }
}
```

tests
```
tests/002-small-simple.txt       timeout 0.5  ternary 3 2 2
tests/073-small-tile.txt         timeout 0.5  tiled 8 8 12 4  
tests/110-medium-tile.txt        timeout 0.5  uniform 200 100 50
tests/116-large-tile.txt         timeout 1.0  tiled 400 400 400 25
```

benchmarks 
```
timeout 3.0   uniform 1000 1000 1000
timeout 3.0   uniform 999 1001 1003  
timeout 23.5  uniform 8000 8000 8000
```

