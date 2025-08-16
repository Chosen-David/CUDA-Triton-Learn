//https://zhuanlan.zhihu.com/p/667521327

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm75.hpp>     // 如果后续涉及 SM75 拷贝行为
#include <cute/util/type_traits.hpp>   // Int<> 等
template<typename T>
__global__ void gemm_simple(T* Cptr,const T* Aptr,const T* Bptr,int m,int n,int k){
    Tensor A=make_tensor(make_gmem_ptr(Aptr),make_shape(m,k),make_stride(k,Int<1>{}));
    Tensor B=make_tensor(make_gmem_ptr(Bptr),make_shape(n,k),make_stride(k,Int<1>{}));
    Tensor C=make_tensor(make_gmem_ptr(Cptr),make_shape(m,n),make_stride(n,Int<1>{}));

}