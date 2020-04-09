#include <torch/extension.h>
// #include <torch/types.h>

#include <vector>

std::vector<torch::Tensor> insert_cuda(torch::Tensor table,
                                       torch::Tensor n_entries, 
                                       torch::Tensor keys,
                                       torch::Tensor hash);


torch::Tensor get_rank_cuda(torch::Tensor table,
                            torch::Tensor keys,
                            torch::Tensor hash);

torch::Tensor get_values_cuda(torch::Tensor table,
                              int n_values);


// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_TYPE(x) AT_ASSERTM(x.dtype()==int64_t, #x " must be a Long Tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> insert(torch::Tensor table,
                                  torch::Tensor n_entries,
                                  torch::Tensor keys,
                                  torch::Tensor hash){
    CHECK_INPUT(table);
    CHECK_INPUT(n_entries);
    CHECK_INPUT(keys);
    CHECK_INPUT(hash);
    return insert_cuda(table, n_entries, keys, hash);        
}

torch::Tensor get_rank(torch::Tensor table,
                       torch::Tensor keys,
                       torch::Tensor hash){
    CHECK_INPUT(table);
    CHECK_INPUT(keys);
    CHECK_INPUT(hash);
    return get_rank_cuda(table, keys, hash);
}

torch::Tensor get_values(torch::Tensor table, int n_values){
  CHECK_INPUT(table);
  return get_values_cuda(table, n_values);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("insert", &insert, "Insert keys into a HashTable");
  m.def("get_rank", &get_rank, "get the rank of queries");
  m.def("get_values", &get_values, "Collect the values of the Hashtable in the rank order.");
}
