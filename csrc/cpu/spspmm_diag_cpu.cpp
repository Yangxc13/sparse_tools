#include "spspmm_diag_cpu.h"
#include "utils.h"

// #include <iostream>
#include <omp.h>
#include <unordered_set>

#define scalar_t float


torch::Tensor
spspmm_diag_sym_AAA_cpu(torch::Tensor rowptr, torch::Tensor col,
                        torch::Tensor weight, int64_t num_threads) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(weight);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(weight.dim() == 1);

  CHECK_INPUT(num_threads >= 1);
  omp_set_num_threads(num_threads);

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto value_diag = torch::zeros({weight.numel()}, weight.options());

  // AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, weight.scalar_type(), "spspmm_diag_sym_AAA", [&] {
    const scalar_t* weight_data = weight.data_ptr<scalar_t>();
    scalar_t* value_data = value_diag.data_ptr<scalar_t>();

    int64_t r = 0; // row, edge, col -> r, e, c
#pragma omp parallel for private(r)
    for (r = 0; r < weight.numel(); r++) {
      scalar_t tmp = 0;
      for (auto e = rowptr_data[r]; e < rowptr_data[r+1]; e++) {
        tmp += weight_data[col_data[e]];
      }
      value_data[r] = weight_data[r] * tmp;
    }
  // });
  return value_diag;
}


torch::Tensor
spspmm_diag_sym_ABA_cpu(torch::Tensor rowptr, torch::Tensor col,
                        torch::Tensor row_weight, torch::Tensor col_weight, int64_t num_threads) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(row_weight);
  CHECK_CPU(col_weight);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(row_weight.dim() == 1);
  CHECK_INPUT(col_weight.dim() == 1);

  CHECK_INPUT(num_threads >= 1);
  omp_set_num_threads(num_threads);

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto value_diag = torch::zeros({row_weight.numel()}, row_weight.options());

  // AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, row_weight.scalar_type(), "spspmm_diag_sym_ABA", [&] {
    const scalar_t* row_weight_data = row_weight.data_ptr<scalar_t>();
    const scalar_t* col_weight_data = col_weight.data_ptr<scalar_t>();
    scalar_t* value_data = value_diag.data_ptr<scalar_t>();

    int64_t r = 0; // row, edge, col -> r, e, c
#pragma omp parallel for private(r)
    for (r = 0; r < row_weight.numel(); r++) {
      scalar_t tmp = 0;
      for (auto e = rowptr_data[r]; e < rowptr_data[r+1]; e++) {
        tmp += col_weight_data[col_data[e]];
      }
      value_data[r] = row_weight_data[r] * tmp;
    }
  // });
  return value_diag;
}


torch::Tensor
spspmm_diag_sym_AAAA_cpu(torch::Tensor rowptr, torch::Tensor col,
                         torch::Tensor weight, int64_t num_threads) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(weight);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(weight.dim() == 1);

  CHECK_INPUT(num_threads >= 1);
  omp_set_num_threads(num_threads);

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto value_diag = torch::zeros({weight.numel()}, weight.options());

  // AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, weight.scalar_type(), "spspmm_diag_sym_AAAA", [&] {
    const scalar_t* weight_data = weight.data_ptr<scalar_t>();
    scalar_t* value_data = value_diag.data_ptr<scalar_t>();

    int64_t r = 0; // row, edge, middle, edge1, col -> r, e0, m, e1, c
#pragma omp parallel for private(r)
    for (r = 0; r < weight.numel(); r++) {
      std::unordered_set<int64_t> neighbors;
      scalar_t tmp0 = 0, tmp1 = 0;
      for (auto e0 = rowptr_data[r]; e0 < rowptr_data[r+1]; e0++) {
        neighbors.emplace(col_data[e0]);
      }
      for (auto e0 = rowptr_data[r]; e0 < rowptr_data[r+1]; e0++) {
        auto m = col_data[e0];
        tmp1 = 0;
        for (auto e1 = rowptr_data[m]; e1 < rowptr_data[m+1]; e1++) {
          auto c = col_data[e1];
          if (neighbors.find(c) != neighbors.end()) {
            tmp1 += weight_data[c];
          }
        }
        tmp0 += weight_data[m] * tmp1;
      }
      value_data[r] = weight_data[r] * tmp0;
    }
  // });
  return value_diag;
}


torch::Tensor
spspmm_diag_ABCA_cpu(torch::Tensor rowptrAB, torch::Tensor colAB, torch::Tensor weightAB,
                     torch::Tensor rowptrBC, torch::Tensor colBC, torch::Tensor weightBC,
                     torch::Tensor rowptrAC, torch::Tensor colAC, torch::Tensor weightCA, int64_t num_threads) {
  CHECK_CPU(rowptrAB);
  CHECK_CPU(colAB);
  CHECK_CPU(weightAB);
  CHECK_CPU(rowptrBC);
  CHECK_CPU(colBC);
  CHECK_CPU(weightBC);
  CHECK_CPU(rowptrAC);
  CHECK_CPU(colAC);
  CHECK_CPU(weightCA);

  CHECK_INPUT(rowptrAB.dim() == 1);
  CHECK_INPUT(colAB.dim() == 1);
  CHECK_INPUT(weightAB.dim() == 1);
  CHECK_INPUT(rowptrBC.dim() == 1);
  CHECK_INPUT(colBC.dim() == 1);
  CHECK_INPUT(weightBC.dim() == 1);
  CHECK_INPUT(rowptrAC.dim() == 1);
  CHECK_INPUT(colAC.dim() == 1);
  CHECK_INPUT(weightCA.dim() == 1);

  CHECK_INPUT(num_threads >= 1);
  omp_set_num_threads(num_threads);

  auto rowptrAB_data = rowptrAB.data_ptr<int64_t>();
  auto colAB_data = colAB.data_ptr<int64_t>();
  auto rowptrBC_data = rowptrBC.data_ptr<int64_t>();
  auto colBC_data = colBC.data_ptr<int64_t>();
  auto rowptrAC_data = rowptrAC.data_ptr<int64_t>();
  auto colAC_data = colAC.data_ptr<int64_t>();

  auto value_diag = torch::zeros({weightAB.numel()}, weightAB.options());

  // AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, weight.scalar_type(), "spspmm_diag_sym_AAAA", [&] {
    const scalar_t *weightAB_data = weightAB.data_ptr<scalar_t>(),
                   *weightBC_data = weightBC.data_ptr<scalar_t>(),
                   *weightCA_data = weightCA.data_ptr<scalar_t>();
    scalar_t* value_data = value_diag.data_ptr<scalar_t>();

    int64_t r = 0; // row, edge, middle, edge1, col -> r, e0, m, e1, c, e2
#pragma omp parallel for private(r)
    for (r = 0; r < weightAB.numel(); r++) {
      std::unordered_set<int64_t> neighbors;
      scalar_t tmp0 = 0, tmp1 = 0;
      for (auto e2 = rowptrAC_data[r]; e2 < rowptrAC_data[r+1]; e2++) {
        neighbors.emplace(colAC_data[e2]);
      }
      for (auto e0 = rowptrAB_data[r]; e0 < rowptrAB_data[r+1]; e0++) {
        auto m = colAB_data[e0];
        tmp1 = 0;
        for (auto e1 = rowptrBC_data[m]; e1 < rowptrBC_data[m+1]; e1++) {
          auto c = colBC_data[e1];
          if (neighbors.find(c) != neighbors.end()) {
            tmp1 += weightCA_data[c];
          }
        }
        tmp0 += weightBC_data[m] * tmp1;
      }
      value_data[r] = weightAB_data[r] * tmp0;
    }
  // });
  return value_diag;
}
