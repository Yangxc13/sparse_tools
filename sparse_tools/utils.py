import datetime
import gc
import numpy as np
import os
import sortednp as snp
from tqdm import tqdm

import torch
from torch_sparse import SparseTensor


class SparseAdjList(object):

    def __init__(self, name, keys=None, adjs=None, num_row=2500604, num_col=21100, with_values=True, root='.', value_type=np.float32):
        self.filename_int64 = f'{root}/{name}_int64.npy'
        self.filename_float32 = f'{root}/{name}_float32.npy'
        self.filename_keys = f'{root}/{name}_keys.pt'
        self.num_row = num_row
        self.num_col = num_col
        self.with_values = with_values
        self.value_type = value_type

        if not os.path.exists(self.filename_int64):
            assert type(adjs) is dict and keys is not None
            if type(keys) != list: keys = list(keys)
            for k, adj in adjs.items():
                if k in keys:
                    assert adj.sizes() == [self.num_row, self.num_col]
            store_adjs = [adjs[k] for k in keys]
            assert len(store_adjs)
            self.write(store_adjs)
            if len(store_adjs) > 1:
                torch.save(keys, self.filename_keys)

        self.fp_int64 = np.memmap(self.filename_int64, dtype=np.int64, mode='r')
        self.num_adjs = self.fp_int64[0]
        assert self.num_row == self.fp_int64[1]
        if self.with_values:
            self.fp_float32 = np.memmap(self.filename_float32, dtype=self.value_type, mode='r')
        if self.num_adjs > 1:
            self.keys = torch.load(self.filename_keys)
            self.r2k = {}
            for i, k in enumerate(self.keys):
                self.r2k[k] = i

        # # passed
        # check_adjs = train_adj_list.load_adjs(expand=True)
        # for i in tqdm(range(self.num_adjs)):
        #     adj0 = adjs[i]
        #     adj1 = check_adjs[i]
        #     assert torch.all(adj0.storage.row() == adj1.storage.row())
        #     assert torch.all(adj0.storage.col() == adj1.storage.col())
        #     if self.with_values:
        #         assert torch.all(adj0.storage.value() == adj1.storage.value())

    def write(self, adjs):
        num_adjs = len(adjs)
        # all_avalidable_nodes_indices = [
        #     torch.where(adj.storage.rowcount() > 0)[0] for adj in tqdm(adjs)]
        # all_avalidable_nodes_enums = [
        #     adj.storage.rowcount()[indices] for indices, adj in tqdm(zip(all_avalidable_nodes_indices, adjs))]

        avaiable_nodes = []
        for adj in adjs:
            avaiable_nodes.append(torch.unique_consecutive(adj.storage.row(), return_counts=True))

        buffer_int64_sizes = [ # avaiable_nodes + rowptr + col
            len(nodes[0])+(len(nodes[0])+1) + adj.nnz() for nodes, adj in tqdm(zip(avaiable_nodes, adjs))]
        buffer_float32_sizes = [ # value
            adj.nnz() for adj in adjs]

        # num_adjs (0) + num_nodes(1) + adjs_offset (2*(num_adjs+1)) + adjs (others)
        file_int64_total_size = 2 + 2 * (num_adjs+1) + sum(buffer_int64_sizes)
        file_float32_total_size = sum(buffer_float32_sizes)

        fp_int64 = np.memmap(self.filename_int64, dtype=np.int64, mode='w+', shape=(file_int64_total_size))
        if self.with_values:
            fp_float32 = np.memmap(self.filename_float32, dtype=self.value_type, mode='w+', shape=(file_float32_total_size))

        ## total num of adjs, addr 0~1
        fp_int64[0] = num_adjs
        ## total num of nodes, addr 1~2
        fp_int64[1] = self.num_row

        ## offset for each adj, addr 2~2*num_adjs+4
        fp_int64[2] = offset = 2 + 2 * (num_adjs+1) # start_bias for adjs in fp_int64
        fp_int64[3] = 0 # # start_bias for adjs in fp_float32

        adjs_int64_offset = np.cumsum(buffer_int64_sizes) + offset
        adjs_float32_offset = np.cumsum(buffer_float32_sizes)

        fp_int64[4:offset] = np.stack((adjs_int64_offset, adjs_float32_offset), axis=1).flatten()

        ## adjs, addr 4*num_adjs+4~end
        for i, adj in tqdm(enumerate(adjs)):
            idx_start, val_start, idx_end, val_end = fp_int64[2+2*i:2+2*i+4]
            num_col = val_end - val_start
            num_nodes = (idx_end - idx_start - num_col) // 2
            num_rowptr = num_nodes + 1
            assert num_nodes + num_rowptr + num_col == idx_end - idx_start

            fp_int64[idx_start:idx_start+num_nodes] = avaiable_nodes[i][0].numpy()
            fp_int64[idx_start+num_nodes] = 0
            fp_int64[idx_start+num_nodes+1:idx_start+num_nodes+num_rowptr] = np.cumsum(avaiable_nodes[i][1].numpy())
            fp_int64[idx_start+num_nodes+num_rowptr:idx_end] = adj.storage.col().data.numpy()
            if self.with_values:
                fp_float32[val_start:val_end] = adj.storage.value().data.numpy()

        assert idx_end == file_int64_total_size
        fp_int64.flush()
        del fp_int64

        if self.with_values:
            assert val_end == file_float32_total_size
            fp_float32.flush()
            del fp_float32

    def load_adj(self, adj_idx=None, adj_key=None, copy=False, expand=False, highlevel=False):
        if adj_idx is not None:
            assert adj_idx < self.num_adjs
        elif adj_key is not None:
            assert adj_key in self.keys
            adj_idx = self.keys.index(adj_key)
        else:
            assert False

        idx_start, val_start, idx_end, val_end = self.fp_int64[2+2*adj_idx:2+2*adj_idx+4]
        num_col = val_end - val_start
        num_nodes = (idx_end - idx_start - num_col) // 2
        num_rowptr = num_nodes + 1
        assert num_nodes + num_rowptr + num_col == idx_end - idx_start

        avaiable_nodes = torch.LongTensor(self.fp_int64[idx_start:idx_start+num_nodes])
        rowptr = torch.LongTensor(self.fp_int64[idx_start+num_nodes:idx_start+num_nodes+num_rowptr])
        col = torch.LongTensor(self.fp_int64[idx_start+num_nodes+num_rowptr:idx_end])
        if self.with_values:
            value=torch.from_numpy(self.fp_float32[val_start:val_end])
        else:
            value = None

        if copy:
            rowptr = rowptr.clone()
            col = col.clone()
            if self.with_values:
                value = value.clone()

        adj = SparseTensor(rowptr=rowptr, col=col, value=value,
                           sparse_sizes=(len(avaiable_nodes), self.num_col), is_sorted=True)

        if highlevel:
            return adj, avaiable_nodes

        if expand:
            row, col, value = adj.coo()
            return SparseTensor(row=avaiable_nodes[row], col=col, value=value,
                                sparse_sizes=(self.num_row, self.num_col), is_sorted=True)
        else:
            return adj

    def load_adjs(self, copy=False, expand=False):
        if self.num_adjs == 1:
            return self.load_adj(0, copy=copy, expand=expand)
        else:
            return {k: self.load_adj(i, copy=copy, expand=expand) for i, k in enumerate(self.keys)}

    def get_rowcounts(self, node_idices, adj_idices):
        # adjs_idx_start_offsets = self.fp_int64[[1+2*adj_idx for adj_idx in adj_idices]]
        # rowptr_offsets = np.array([(offset+node_idx, offset+node_idx+1)
        #     for offset, node_idx in zip(adjs_idx_start_offsets, node_idices)])
        # rowptrs = self.fp_int64[rowptr_offsets]
        # return rowptrs[:,1] - rowptrs[:,0]
        # ---
        # # tic = datetime.datetime.now()
        # start_ends = self.fp_int64[2:2+2*(self.num_adjs+1)].copy()
        # out = []
        # for node_idx, adj_idx in zip(node_idices, adj_idices):
        #     idx_start, val_start, idx_end, val_end = start_ends[2*adj_idx:2*adj_idx+4]
        #     num_col = val_end - val_start
        #     num_nodes = (idx_end - idx_start - num_col) // 2
        #     num_rowptr = num_nodes + 1
        #     assert num_nodes + num_rowptr + num_col == idx_end - idx_start

        #     avaiable_nodes = self.fp_int64[idx_start:idx_start+num_nodes]
        #     node_pos = avaiable_nodes.searchsorted(node_idx)
        #     if node_pos == num_nodes or avaiable_nodes[node_pos] != node_idx:
        #         out.append(0)
        #     else:
        #         rowptr_offsets = idx_start + num_nodes + node_pos
        #         rowptr = self.fp_int64[rowptr_offsets:rowptr_offsets+2]
        #         out.append(rowptr[1] - rowptr[0])
        # out = np.array(out)
        # # toc = datetime.datetime.now()
        # # print(toc-tic)
        # return out
        # ---
        # tic = datetime.datetime.now()
        start_ends = self.fp_int64[2:2+2*(self.num_adjs+1)].copy()

        idx_starts, val_starts, idx_ends, val_ends = start_ends[2*adj_idices.reshape((1,-1))+np.arange(4).reshape((-1,1))]
        num_cols = val_ends - val_starts
        num_nodes = (idx_ends - idx_starts - num_cols) // 2
        num_rowptrs = num_nodes + 1
        assert np.all(num_nodes + num_rowptrs + num_cols == idx_ends - idx_starts)

        avaiable_nodes = self.fp_int64[np.concatenate([
            np.arange(idx_start, idx_start+num) for idx_start, num in zip(idx_starts, num_nodes)])]
        avaiable_nodes = np.split(avaiable_nodes, np.cumsum(num_nodes)[:-1])

        col_num = []
        rowptr_offset_indices = []
        for i, (node_idx, nodes) in enumerate(zip(node_idices, avaiable_nodes)):
            node_pos = nodes.searchsorted(node_idx)
            if node_pos == len(nodes) or nodes[node_pos] != node_idx:
                col_num.append(0)
            else:
                col_num.append(1)
                rowptr_offset_indices.append(idx_starts[i] + num_nodes[i] + node_pos)
        col_num = np.array(col_num, dtype=np.int64)
        rowptr_offset_indices = np.array(rowptr_offset_indices)

        rowptr_offsets = self.fp_int64[np.stack((rowptr_offset_indices, rowptr_offset_indices+1), axis=1)]
        col_num[col_num > 0] = rowptr_offsets[:,1] - rowptr_offsets[:,0]
        # toc = datetime.datetime.now()
        # print(toc-tic)
        return col_num

    def get_cols(self, node_idices, adj_idices):
        start_ends = self.fp_int64[2:2+2*(self.num_adjs+1)].copy()

        idx_starts, val_starts, idx_ends, val_ends = start_ends[2*adj_idices.reshape((1,-1))+np.arange(4).reshape((-1,1))]
        num_cols = val_ends - val_starts
        num_nodes = (idx_ends - idx_starts - num_cols) // 2
        num_rowptrs = num_nodes + 1
        assert np.all(num_nodes + num_rowptrs + num_cols == idx_ends - idx_starts)

        avaiable_nodes = self.fp_int64[np.concatenate([
            np.arange(idx_start, idx_start+num) for idx_start, num in zip(idx_starts, num_nodes)])]
        avaiable_nodes = np.split(avaiable_nodes, np.cumsum(num_nodes)[:-1])

        col_num = []
        rowptr_offset_indices = []
        for i, (node_idx, nodes) in enumerate(zip(node_idices, avaiable_nodes)):
            node_pos = nodes.searchsorted(node_idx)
            if node_pos == len(nodes) or nodes[node_pos] != node_idx:
                col_num.append(0)
                rowptr_offset_indices.append(-1)
            else:
                col_num.append(1)
                rowptr_offset_indices.append(idx_starts[i] + num_nodes[i] + node_pos)
        col_num = np.array(col_num, dtype=np.int64)
        rowptr_offset_indices = np.array(rowptr_offset_indices)

        rowptr_offsets = self.fp_int64[np.stack((rowptr_offset_indices, rowptr_offset_indices+1), axis=1)]
        col_num[col_num > 0] = (rowptr_offsets[:,1] - rowptr_offsets[:,0])[col_num > 0]
        # ---

        col_indices = []
        for (rowptr_start, rowptr_end), idx_start, num, num_rowptr in zip(rowptr_offsets, idx_starts, num_nodes, num_rowptrs):
            if rowptr_start >= 0:
                col_indices.append(np.arange(rowptr_start, rowptr_end) + idx_start + num + num_rowptr)
        col_indices = np.concatenate(col_indices)
        cols = np.split(self.fp_int64[col_indices], np.cumsum(col_num)[:-1])
        return cols

    def load_batch(self, nodes, avoid_triplets=None, sample_metapath_ratio=1., sample_neighbors=-1):
        if isinstance(nodes, torch.LongTensor):
            nodes = nodes.numpy()

        _tmp = {}
        if avoid_triplets is not None:
            if isinstance(nodes, torch.LongTensor):
                avoid_triplets = avoid_triplets.numpy()
            for h, r, t in avoid_triplets:
                if r not in _tmp: _tmp[r] = {}
                if t not in _tmp[r]:
                    _tmp[r][t] = [h]
                elif h not in _tmp[r][t]:
                    _tmp[r][t].append(h)

        rows, values, adj_ids = [], [], []
        start_ends = self.fp_int64[2:2+2*(self.num_adjs+1)].copy()

        for i in range(self.num_adjs):
            # tic = datetime.datetime.now()
            idx_start, val_start, idx_end, val_end = start_ends[2*i:2*i+4]
            num_col = val_end - val_start
            num_nodes = (idx_end - idx_start - num_col) // 2
            num_rowptr = num_nodes + 1
            assert num_nodes + num_rowptr + num_col == idx_end - idx_start

            all_nodes = self.fp_int64[idx_start:idx_start+num_nodes]
            _, (nodes_offset, avaiable_nodes) = snp.intersect(all_nodes, nodes, indices=True)
            if len(avaiable_nodes) == 0:
                continue

            if sample_metapath_ratio < 1.:
                mask = torch.ones(len(avaiable_nodes))
                mask = F.dropout(mask, p=1-sample_metapath_ratio) > 0

                if torch.any(mask):
                    # _ = _[mask.numpy()]
                    nodes_offset = nodes_offset[mask.numpy()]
                    avaiable_nodes = avaiable_nodes[mask.numpy()]
                else:
                    continue

            ptr_slices = np.stack((idx_start+num_nodes+nodes_offset, idx_start+num_nodes+nodes_offset+1), axis=1)
            ptr_start_ends = self.fp_int64[ptr_slices]

            rowptr = np.r_[[0], ptr_start_ends[:,1] - ptr_start_ends[:,0]]
            rowptr = np.cumsum(rowptr)

            col_slices = np.concatenate([
                np.arange(idx_start+num_nodes+num_rowptr+ptr_start, idx_start+num_nodes+num_rowptr+ptr_end)
                for ptr_start, ptr_end in ptr_start_ends])

            if self.with_values:
                value_slices = np.concatenate([
                    np.arange(val_start+ptr_start, val_start+ptr_end)
                    for ptr_start, ptr_end in ptr_start_ends])
                value = torch.from_numpy(self.fp_float32[value_slices].copy())
            else:
                value = None

            tmp = SparseTensor(
                rowptr=torch.LongTensor(rowptr),
                col=torch.LongTensor(self.fp_int64[col_slices].copy()), value=value,
                sparse_sizes=(len(avaiable_nodes), self.num_col), is_sorted=True)

            if i in _tmp: # for avoid_triplets
                n0, (n1, n2) = snp.intersect(np.array(list(_tmp[i].keys())), avaiable_nodes, indices=True)
                if len(n0):
                    mask = torch.ones(tmp.nnz(), dtype=torch.bool)
                    for k, idx in zip(n2, n0):
                        vs = _tmp[i][idx]
                        k_offset_s, k_offset_e = tmp.storage._rowptr[k], tmp.storage._rowptr[k+1]
                        flag = False
                        for v in vs:
                            v_idx = np.searchsorted(tmp.storage._col[k_offset_s:k_offset_e].numpy(), v)
                            if v_idx < k_offset_e - k_offset_s:
                                flag = True
                                mask[k_offset_s + v_idx] = 0
                                tmp.storage._value[k_offset_s + v_idx] = 0
                        if flag:
                            tmp.storage._value[k_offset_s:k_offset_e] /= (tmp.storage._value[k_offset_s:k_offset_e].sum() + 1e-12)

                    if torch.any(~mask):
                        tmp = tmp.masked_select_nnz(mask, layout='coo')
                        if tmp.nnz() == 0: continue
                        mask = tmp.storage.rowcount() > 0
                        if torch.any(~mask):
                            avaiable_nodes = avaiable_nodes[mask]
                            tmp = tmp[mask]

            if sample_neighbors > 0 and tmp.storage.rowcount().max() > sample_neighbors:
                tmp_adj, tmp_nid = tmp.sample_adj(torch.arange(tmp.size(0)), sample_neighbors)
                tmp = SparseTensor(
                    rowptr=tmp_adj.storage.rowptr(),
                    col=tmp_nid[tmp_adj.storage.col()],
                    value=tmp_adj.storage.value(), sparse_sizes=tmp.sizes())
                if tmp.storage.value() is not None:
                    tmp.storage._value = tmp.storage.value() / tmp.sum(dim=1)[tmp.storage.row()]

            # if sample_metapath_ratio < 1.:
            #     mask = torch.ones(len(avaiable_nodes))
            #     mask = F.dropout(mask, p=1-sample_metapath_ratio) > 0

            #     if torch.any(mask):
            #         avaiable_nodes = avaiable_nodes[mask.numpy()]
            #         tmp = tmp[mask]
            #     else:
            #         continue

            rows.append(avaiable_nodes)
            values.append(tmp)
            adj_ids.append(i)
            # toc = datetime.datetime.now()
            # print(i, toc-tic, start_ends[2*i+2]-start_ends[2*i+4])

        return rows, values, adj_ids
