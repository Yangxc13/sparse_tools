#pragma once

#include <torch/extension.h>


torch::Tensor
spspmm_diag_sym_AAA_cpu(torch::Tensor rowptr, torch::Tensor col,
                        torch::Tensor weight, int64_t num_threads);


torch::Tensor
spspmm_diag_sym_ABA_cpu(torch::Tensor rowptr, torch::Tensor col,
                        torch::Tensor row_weight, torch::Tensor col_weight, int64_t num_threads);


torch::Tensor
spspmm_diag_sym_AAAA_cpu(torch::Tensor rowptr, torch::Tensor col,
                         torch::Tensor weight, int64_t num_threads);


torch::Tensor
spspmm_diag_ABCA_cpu(torch::Tensor rowptrAB, torch::Tensor colAB, torch::Tensor weightAB,
                     torch::Tensor rowptrBC, torch::Tensor colBC, torch::Tensor weightBC,
                     torch::Tensor rowptrAC, torch::Tensor colAC, torch::Tensor weightCA, int64_t num_threads);