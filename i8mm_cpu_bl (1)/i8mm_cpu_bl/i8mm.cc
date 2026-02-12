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


