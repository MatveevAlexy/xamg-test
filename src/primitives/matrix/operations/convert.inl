/****************************************************************************
** 
**  Copyright (C) 2019-2021 Boris Krasnopolsky, Alexey Medvedev
**  Contact: xamg-test@imec.msu.ru
** 
**  This file is part of the XAMG library.
** 
**  Commercial License Usage
**  Licensees holding valid commercial XAMG licenses may use this file in
**  accordance with the terms of commercial license agreement.
**  The license terms and conditions are subject to mutual agreement
**  between Licensee and XAMG library authors signed by both parties
**  in a written form.
** 
**  GNU General Public License Usage
**  Alternatively, this file may be used under the terms of the GNU
**  General Public License, either version 3 of the License, or (at your
**  option) any later version. The license is as published by the Free 
**  Software Foundation and appearing in the file LICENSE.GPL3 included in
**  the packaging of this file. Please review the following information to
**  ensure the GNU General Public License requirements will be met:
**  https://www.gnu.org/licenses/gpl-3.0.html.
** 
****************************************************************************/

#pragma once

namespace XAMG {
namespace matrix {

template <typename F0, typename I01, typename I02, typename I03, typename I04, typename F,
          typename I1, typename I2, typename I3, typename I4>
void convert(const csr_matrix<F0, I01, I02, I03, I04> &mat_in,
             csr_matrix<F, I1, I2, I3, I4> &mat_out) {
    mat_out.nrows = mat_in.nrows;
    mat_out.ncols = mat_in.ncols;
    mat_out.nonzeros = mat_in.nonzeros;

    mat_out.block_nrows = mat_in.block_nrows;
    mat_out.block_ncols = mat_in.block_ncols;

    mat_out.block_row_offset = mat_in.block_row_offset;
    mat_out.block_col_offset = mat_in.block_col_offset;

    mat_out.if_indexed = mat_in.if_indexed;

    if (mat_out.sharing_mode == mem::NUMA) {
        mpi::bcast<uint64_t>(&mat_out.nrows, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.ncols, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.nonzeros, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.block_nrows, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.block_ncols, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.block_row_offset, 1, 0, mpi::INTRA_NUMA);
        mpi::bcast<uint64_t>(&mat_out.block_col_offset, 1, 0, mpi::INTRA_NUMA);
        uint8_t _ind = mat_out.if_indexed;
        mpi::bcast<uint8_t>(&_ind, 1, 0, mpi::INTRA_NUMA);
        mat_out.if_indexed = _ind;
    }

    mat_out.alloc();

    /////////

    if (!mat_out.if_empty) {
        vector::convert<I01, I1>(mat_in.row, mat_out.row);
        vector::convert<I02, I2>(mat_in.col, mat_out.col);
        vector::convert<F0, F>(mat_in.val, mat_out.val);

        if (mat_out.sharing_mode == mem::NUMA) {
            bcast_vector_state(mat_out.row, mpi::INTRA_NUMA);
            bcast_vector_state(mat_out.col, mpi::INTRA_NUMA);
            bcast_vector_state(mat_out.val, mpi::INTRA_NUMA);
        }

        if (mat_out.if_indexed) {
            vector::convert<I03, I3>(mat_in.row_ind, mat_out.row_ind);
            vector::convert<I04, I4>(mat_in.col_ind, mat_out.col_ind);

            if (mat_out.sharing_mode == mem::NUMA) {
                bcast_vector_state(mat_out.row_ind, mpi::INTRA_NUMA);
                bcast_vector_state(mat_out.col_ind, mpi::INTRA_NUMA);
            }
        }
    }
}

template <typename F0, typename F>
void convert(const dense_matrix<F0> &mat_in, dense_matrix<F> &mat_out) {
    mat_out.nrows = mat_in.nrows;
    mat_out.ncols = mat_in.ncols;

    mat_out.block_nrows = mat_in.block_nrows;
    mat_out.block_ncols = mat_in.block_ncols;

    mat_out.block_row_offset = mat_in.block_row_offset;
    mat_out.block_col_offset = mat_in.block_col_offset;

    mat_out.if_indexed = mat_in.if_indexed;

    mat_out.alloc();

    /////////

    if (!mat_out.if_empty) {
        vector::convert<F0, F>(mat_in.val, mat_out.val);

        if (mat_out.if_indexed) {
            vector::convert<uint32_t, uint32_t>(mat_in.row_ind, mat_out.row_ind);
            vector::convert<uint32_t, uint32_t>(mat_in.col_ind, mat_out.col_ind);
        }
    }
}

template <typename F, typename I1, typename I2, typename I3, typename I4>
void unpack_to_csr(const matrix &m, csr_matrix<F, I1, I2, I3, I4> &csr) {
    auto &numa_layer = m.data_layer.find(segment::NUMA)->second;
    auto &node_layer = m.data_layer.find(segment::NODE)->second;

    uint32_t core_size = m.row_part->core_layer.block_size[id.nm_core];
    uint32_t core_offset = m.row_part->core_layer.block_indx[id.nm_core];

    csr.nrows = m.row_part->core_layer.block_size[id.nm_core];
    csr.ncols = m.col_part->node_layer.block_indx.back();
    csr.block_nrows = csr.nrows;
    csr.block_ncols = csr.ncols;
    csr.block_row_offset = numa_layer.diag.data->block_row_offset + core_offset;
    csr.block_col_offset = 0;

    csr.nonzeros = numa_layer.diag.data->get_range_size(core_offset, core_offset + core_size);
    for (const auto &offd : numa_layer.offd)
        csr.nonzeros +=
            offd.data->get_uncompressed_range_size(core_offset, core_offset + core_size);
    for (const auto &offd : node_layer.offd)
        csr.nonzeros +=
            offd.data->get_uncompressed_range_size(core_offset, core_offset + core_size);

    csr.alloc();

    if (csr.if_empty)
        return;

    auto row_ptr = csr.row.template get_aligned_ptr<I1>();
    auto col_ptr = csr.col.template get_aligned_ptr<I2>();
    auto val_ptr = csr.val.template get_aligned_ptr<F>();
    row_ptr[0] = 0;
    ////

    std::vector<int> col;
    std::vector<float64_t> val;

    uint64_t row_offset = numa_layer.diag.data->get_block_row_offset();
    uint64_t col_offset = numa_layer.diag.data->get_block_col_offset();

    std::vector<uint32_t> numa_offd_indx(numa_layer.offd.size(), 0);
    std::vector<uint32_t> node_offd_indx(node_layer.offd.size(), 0);

    for (uint32_t l = core_offset; l < core_offset + core_size; ++l) {
        int row_size = numa_layer.diag.data->get_row_size(l);
        numa_layer.diag.data->get_row(l, col, val);

        int ll = l + row_offset;
        for (uint32_t ii = 0; ii < (uint32_t)row_size; ++ii)
            col[ii] += col_offset;

        std::vector<int> col0;
        std::vector<float64_t> val0;

        for (uint32_t nb = 0; nb < numa_layer.offd.size(); ++nb) {
            if (numa_layer.offd[nb].data->if_empty)
                continue;
            numa_layer.offd[nb].data->get_row_and_unpack(l, col0, val0, numa_offd_indx[nb]);
            for (uint32_t i = 0; i < col0.size(); ++i) {
                col.push_back(col0[i]);
                val.push_back(val0[i]);
            }
        }

        for (uint32_t nb = 0; nb < node_layer.offd.size(); ++nb) {
            if (node_layer.offd[nb].data->if_empty)
                continue;
            node_layer.offd[nb].data->get_row_and_unpack(l, col0, val0, node_offd_indx[nb]);
            for (uint32_t i = 0; i < col0.size(); ++i) {
                col.push_back(col0[i]);
                val.push_back(val0[i]);
            }
        }

        // sort();
        for (uint32_t iii = 0; iii < col.size(); ++iii) {
            for (uint32_t jjj = iii + 1; jjj < col.size(); ++jjj) {
                if (col[iii] > col[jjj]) {
                    std::swap(col[iii], col[jjj]);
                    std::swap(val[iii], val[jjj]);
                }
            }
        }

        uint32_t iii = l - core_offset;
        for (size_t ll = 0; ll < col.size(); ++ll) {
            col_ptr[row_ptr[iii] + ll] = col[ll];
            val_ptr[row_ptr[iii] + ll] = val[ll];
        }
        row_ptr[iii + 1] = row_ptr[iii] + col.size();
    }
    assert(csr.nonzeros == row_ptr[core_size]);

    csr.row.if_zero = false;
    csr.row.if_initialized = true;
    csr.col.if_zero = false;
    csr.col.if_initialized = true;
    csr.val.if_zero = false;
    csr.val.if_initialized = true;
}

} // namespace matrix
} // namespace XAMG
