#include <cublas_v2.h>
#include <cuda.h> // NOLINT
#include <cute/tensor.hpp>
#include <stdlib.h>

using namespace cute;
//z=a*x+b*y+c
template<int kNUmElemPerThread=8>
__global__ void add_vec_cute(
    half *z, int num, const half *x, const half *y, const half a, const half b,
    const half c) {
        int idx=threadIdx.x+blockDim.x*blockIdx.x;
        if(idx>=num/kNUmElemPerThread){
            return;
        }
        Tensor tx=make_tensor(make_gmem_ptr(x),make_shape(num));
        Tensor ty=make_tensor(make_gmem_ptr(y),make_shape(num));
        Tensor tz=make_tensor(make_gmem_ptr(z),make_shape(num));
        
        Tensor txr=local_tile(tx,make_shape(Int<kNUmElemPerThread>{}),make_coord(idx)); 
        Tensor tyr=local_tile(ty,make_shape(Int<kNUmElemPerThread>{}),make_coord(idx)); 
        Tensor tzr=local_tile(tz,make_shape(Int<kNUmElemPerThread>{}),make_coord(idx)); 

        Tensor txR=make_tensor_like(txr);
        Tensor tyR=make_tensor_like(tyr);
        Tensor tzR=make_tensor_like(tzr);

        copy(txr,txR);
        copy(tyr,tyR);
        copy(tzr,tzR);

        half2 a2={a,a};
        half2 b2={b,b};
        half2 c2={c,c};

        auto txR2=recast<half2>(txR);
        auto tyR2=recast<half2>(tyR);
        auto tzR2=recast<half2>(tzR);

        #pragma unroll
        for(int i=0;i<tzR2.size();i++){
            //这样就可以显式优化成两条FMA指令了，hfma2。
            tzR2(i)=a2*txR2(i)+(b2*tyR2(i)+c2);
        }

        auto tzR2_res=recast<half>(tzR2);
        copy(tzR2_res,tzr);
}