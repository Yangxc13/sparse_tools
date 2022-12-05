import torch
from torch_sparse import SparseTensor


# symmetric: A^T == A
def spspmm_diag_sym_AAA(src: SparseTensor, num_threads: int = 4) -> torch.Tensor:
    assert src.sparse_size(0) == src.sparse_size(1)
    # assert not check or (torch.all(src.get_diag() == 0) and src.is_symmetric())
    rowptr, col, _ = src.csr()
    weight = 1. / src.storage.rowcount()
    weight[torch.isnan(weight) | torch.isinf(weight)] = 0.

    return torch.ops.sparse_tools.spspmm_diag_sym_AAA(rowptr, col, weight, num_threads)


# symmetric: A^T == B
def spspmm_diag_sym_ABA(src: SparseTensor, colcount: torch.Tensor = None, num_threads: int = 4) -> torch.Tensor:
    rowptr, col, _ = src.csr()
    row_weight = 1. / src.storage.rowcount()
    row_weight[torch.isnan(row_weight) | torch.isinf(row_weight)] = 0.
    if colcount is None:
        colcount = src.storage.colcount()
    col_weight = 1. / colcount
    col_weight[torch.isnan(col_weight) | torch.isinf(col_weight)] = 0.

    return torch.ops.sparse_tools.spspmm_diag_sym_ABA(rowptr, col, row_weight, col_weight, num_threads)


# symmetric: A^T == A
def spspmm_diag_sym_AAAA(src: SparseTensor, num_threads: int = 4) -> torch.Tensor:
    assert src.sparse_size(0) == src.sparse_size(1)
    # assert not check or (torch.all(src.get_diag() == 0) and src.is_symmetric())
    rowptr, col, _ = src.csr()
    weight = 1. / src.storage.rowcount()
    weight[torch.isnan(weight) | torch.isinf(weight)] = 0.

    return torch.ops.sparse_tools.spspmm_diag_sym_AAAA(rowptr, col, weight, num_threads)


def spspmm_diag_ABCA(AB: SparseTensor, BC: SparseTensor, AC: SparseTensor,
                     colcountAC: torch.Tensor = None, num_threads: int = 4) -> torch.Tensor:
    assert AB.sparse_size(0) == AC.sparse_size(0)
    assert AB.sparse_size(1) == BC.sparse_size(0)
    assert BC.sparse_size(1) == AC.sparse_size(1)

    rowptrAB, colAB, _ = AB.csr()
    weightAB = 1. / AB.storage.rowcount()
    weightAB[torch.isnan(weightAB) | torch.isinf(weightAB)] = 0.

    rowptrBC, colBC, _ = BC.csr()
    weightBC = 1. / BC.storage.rowcount()
    weightBC[torch.isnan(weightBC) | torch.isinf(weightBC)] = 0.

    rowptrAC, colAC, _ = AC.csr()
    if colcountAC is None:
        colcountAC = AC.storage.colcountAC()
    weightCA = 1. / colcountAC
    weightCA[torch.isnan(weightCA) | torch.isinf(weightCA)] = 0.

    return torch.ops.sparse_tools.spspmm_diag_ABCA(rowptrAB, colAB, weightAB,
        rowptrBC, colBC, weightBC, rowptrAC, colAC, weightCA, num_threads);
