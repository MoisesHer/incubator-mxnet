/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2015 by Contributors
 * \file matrix_op.cu
 * \brief GPU Implementation of matrix operations
 */
#include <cub/cub.cuh>
#include "./matrix_op-inl.h"
#include "./elemwise_unary_op.h"


namespace mxnet {
namespace op {

/*!
 * \brief Compute the number of elements of every row.
 */
struct SliceMarkCsrIndPtr {
  /*!
   * \brief
   * \param i           the i-th row of the output csr ndarray
   * \param prefix_sum  indptr array of the output csr ndarray
   * \param in_idx      indices array of the input csr ndarray
   * \param in_indptr   indptr array of the input csr ndarray
   * \param begin_col   starting indice
   * \param end_col     ending indice
   */
  template<typename IType, typename RType>
  MSHADOW_XINLINE static void Map(int i,
                                  RType* prefix_sum,
                                  const IType* in_idx,
                                  const RType* in_indptr,
                                  const int begin_col, const int end_col) {
    if (i == 0) {
      prefix_sum[0] = 0;
    }
    RType size = 0;
    for (RType j = in_indptr[i]; j < in_indptr[i+1]; j++) {
      // indices of CSRNDArray are in ascending order per row
      if (in_idx[j] >= end_col) {
        break;
      } else if (in_idx[j] >= begin_col) {
        size++;
      }
    }
    prefix_sum[i+1] = size;
  }
};


template<>
void SliceDimTwoCsrImpl<gpu>(const mxnet::TShape &begin, const mxnet::TShape &end,
                             const OpContext& ctx, const NDArray &in, const NDArray &out) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;

  Stream<gpu> *s = ctx.get_stream<gpu>();

  nnvm::dim_t begin_row = begin[0], end_row = end[0];
  nnvm::dim_t begin_col = begin[1], end_col = end[1];
  nnvm::dim_t indptr_len = end_row - begin_row + 1;
  out.CheckAndAllocAuxData(kIndPtr, Shape1(indptr_len));
  // assume idx indptr share the same type
  MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIndPtr), RType, {
    MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIdx), IType, {
      MSHADOW_TYPE_SWITCH(in.dtype(), DType, {
        RType *in_indptr = in.aux_data(kIndPtr).dptr<RType>();
        IType *in_idx = in.aux_data(kIdx).dptr<IType>();
        DType *in_data = in.data().dptr<DType>();

        RType *out_indptr = out.aux_data(kIndPtr).dptr<RType>();

        Kernel<SliceMarkCsrIndPtr, gpu>::Launch(s, indptr_len - 1,
                                                out_indptr,
                                                in_idx,
                                                in_indptr + begin_row,
                                                begin_col, end_col);
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      out_indptr,
                                      out_indptr,
                                      indptr_len,
                                      Stream<gpu>::GetStream(s));
        Tensor<gpu, 1, char> workspace = ctx.requested[0]
            .get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
        d_temp_storage = workspace.dptr_;

        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      out_indptr,
                                      out_indptr,
                                      indptr_len,
                                      Stream<gpu>::GetStream(s));
        // retrieve nnr
        RType nnr = 0;
        CUDA_CALL(cudaMemcpyAsync(&nnr, &out_indptr[indptr_len-1], sizeof(RType),
                                  cudaMemcpyDeviceToHost, mshadow::Stream<gpu>::GetStream(s)));
        CUDA_CALL(cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s)));

        // returns zeros in csr format if nnr = 0
        if (nnr == 0) {
          out.set_aux_shape(kIdx, Shape1(0));
          return;
        }
        out.CheckAndAllocAuxData(kIdx, Shape1(nnr));
        out.CheckAndAllocData(Shape1(nnr));
        IType *out_idx = out.aux_data(kIdx).dptr<IType>();
        DType *out_data = out.data().dptr<DType>();

        Kernel<SliceDimTwoCsrAssign, gpu>::Launch(s, indptr_len - 1, out_idx, out_data,
                                                  out_indptr, in_idx, in_data,
                                                  in_indptr + begin_row,
                                                  begin_col, end_col);
      });
    });
  });
}

constexpr size_t size2activate_join_sectors = 32;
constexpr size_t split_max_sections = 82;
template <typename DType>
struct split_tensor_data {
  size_t num_sections;
  DType* inputs[1];
  DType* outputs[split_max_sections];
  size_t in_strides[split_max_sections];
  size_t out_strides[split_max_sections];
  size_t n_elements[split_max_sections];
  size_t original_section_sizes[split_max_sections];
  size_t accum_elems_block[split_max_sections];
};
   
template <bool several_sections_per_block, typename LType, typename DType>
__global__ void split_tensor_kernel(const split_tensor_data<DType> params,
                                    size_t total_elems_split_dim,
                                    size_t n_blocks_split_dim,
                                    size_t n_sections_block,
                                    size_t leading_size,
                                    size_t n_iters_leading_dims) {
  const int entries_per_load = sizeof(LType)/sizeof(DType);
  extern __shared__ int sharedmem[];
  DType* elements =  reinterpret_cast<DType*>(sharedmem);

  if (several_sections_per_block) {
    // M sections, N blocks, ignores LoadType
    // assumes (elements assigned to block) < (threads within block)
    size_t start_sector = (blockIdx.x % n_blocks_split_dim) * n_sections_block;
    size_t start_pos_in = params.in_strides[start_sector];
    size_t last_sector = std::min(start_sector + n_sections_block - 1, params.num_sections - 1);
    size_t end_pos_in = params.in_strides[last_sector] + params.n_elements[last_sector] - 1;
    size_t n_elements = end_pos_in - start_pos_in + 1;

    // Binary search to find sector of each thread
    size_t lower = start_sector;
    size_t upper = last_sector;
    while (lower < upper) {
      size_t mid = (lower + upper + 1) / 2;
      if (threadIdx.x >= params.accum_elems_block[mid])
        lower = mid;
      else
        upper = mid - 1;
    }
    size_t my_sector = upper;
    size_t n_elems_my_sector = params.n_elements[my_sector];

    size_t leading_offset_in = size_t(blockIdx.x / n_blocks_split_dim) *
                               total_elems_split_dim * n_iters_leading_dims;
    size_t n_iters = n_iters_leading_dims;
    if ((size_t(blockIdx.x / n_blocks_split_dim) + 1) * n_iters_leading_dims >
        leading_size) {
      n_iters = leading_size - size_t(blockIdx.x / n_blocks_split_dim) *
                n_iters_leading_dims;
    }                       
    // read elements for several sectors into shared, iterating over split dim
    for (size_t iter = 0; iter < n_iters; ++iter) {
      if (threadIdx.x < n_elements) {
        // elements in shared follow pattern: Section-0{0..N}, Section-1{0..N}, etc..
        size_t pos_in_shared = ((my_sector - start_sector) * size2activate_join_sectors) +
                               (iter * n_elems_my_sector) +
                                threadIdx.x - params.accum_elems_block[my_sector];
        elements[pos_in_shared] = params.inputs[0][leading_offset_in + start_pos_in +
                                                   iter * total_elems_split_dim +
                                                   threadIdx.x];
      }
    }
    // write into each section
    size_t offset_shared = 0;
    for (size_t sec = start_sector; sec <= last_sector; ++sec) {
      size_t leading_offset_out = size_t(blockIdx.x / n_blocks_split_dim) *
                                  n_iters_leading_dims * params.original_section_sizes[sec];
      for (size_t i = threadIdx.x; i < params.n_elements[sec] * n_iters; i += blockDim.x) {
        size_t offset, offset_remaining_elems_sector;
        if (params.n_elements[sec] < params.original_section_sizes[sec]) {
          // if there are remanining elems for this sector on this axis 
          // (because one section was subdivided further): then can be uncoalesced accesses
          // this only happens with irregular sections sizes & small sections
          offset_remaining_elems_sector = size_t(i / params.n_elements[sec]) *
                                          params.original_section_sizes[sec];
          offset = size_t(i % params.n_elements[sec]);
        } else {
          offset_remaining_elems_sector = 0;
          offset = i;
        }
        params.outputs[sec][leading_offset_out + params.out_strides[sec] +
                            offset_remaining_elems_sector + offset] =
            elements[offset_shared + i];
          
      }
      offset_shared += size2activate_join_sectors;
    }
    
  } else {
    // 1 section, N blocks
    size_t section_id = blockIdx.x % params.num_sections;
    size_t n_elements =  entries_per_load > 0 ?
                         params.n_elements[section_id] / entries_per_load : 0;
    const LType* in_aligned = reinterpret_cast<const LType*>(params.inputs[0]);
    LType* out_aligned = reinterpret_cast<LType*>(params.outputs[section_id]);
    size_t leading_offset_in = entries_per_load ?
                               int(blockIdx.x / params.num_sections) *
                               total_elems_split_dim / entries_per_load : 0;
    size_t leading_offset_out = entries_per_load ? 
                                int(blockIdx.x / params.num_sections) *
                                params.original_section_sizes[section_id] / entries_per_load : 0;
    for (size_t i = threadIdx.x; i < n_elements; i += blockDim.x) {
      size_t offset_in = entries_per_load ?
                         params.in_strides[section_id] / entries_per_load + i : 0;
      size_t offset_out = entries_per_load ?
                          params.out_strides[section_id] / entries_per_load + i : 0;
      LType e = in_aligned[leading_offset_in + offset_in];
      out_aligned[leading_offset_out + offset_out] = e;
    }
  }
}

template <typename DType>
int get_load_type_split(const split_tensor_data<DType> params) {
  using namespace mshadow;
  int sectors_largest_multiple = 8;
  for (size_t i = 0; i < params.num_sections; ++i) {
    if (params.n_elements[i] * sizeof(DType) % 8)
      sectors_largest_multiple = std::min(sectors_largest_multiple, 4);
    if (params.n_elements[i] * sizeof(DType) % 4)
      sectors_largest_multiple = std::min(sectors_largest_multiple, 2);
    if (params.n_elements[i] * sizeof(DType) % 2)
      sectors_largest_multiple = std::min(sectors_largest_multiple, 1);
  }
  if (sectors_largest_multiple == 8) {
    return kFloat64;
  } else if (sectors_largest_multiple >= 4) {
    return kFloat32;
  } else if (sectors_largest_multiple >= 2) {
    return kFloat16;
  } else {
    return kUint8;
  }
  return kUint8;
}    

template<>
inline void SplitOpForwardImpl<gpu>(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs,
                                    const int real_axis) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  const SplitParam& param = nnvm::get<SplitParam>(attrs.parsed);
  Stream<gpu> *s = ctx.get_stream<gpu>();
  const TBlob& input_data = inputs[split_enum::kData];
  CHECK_LT(real_axis, input_data.ndim());
  const mxnet::TShape& ishape = input_data.shape_;
  const mxnet::TShape split_pts =
    (param.sections > 0) ? GetSplitIndices(ishape, real_axis, param.sections) : param.indices;
  std::vector<size_t> indices;
  
  for (const auto& split_pos : split_pts) {
    indices.push_back(split_pos);
  }
  if (param.sections == 0) {
    indices.push_back(ishape[real_axis]);
  }
  int original_n_sections = indices.size() - 1;

  size_t tail_size = 1;
  for (int i = real_axis + 1; i < input_data.ndim(); ++i) {
    tail_size *= input_data.shape_[i];
  }
  size_t leading_size = 1;
  for (int i = 0; i < real_axis; ++i) {
    leading_size *= input_data.shape_[i];
  }

  size_t smallest_section_size = (indices[1] - indices[0]) * tail_size;
  for (int i=0; i < (indices.size() -1); ++i) {
    size_t section_size = (indices[i+1] - indices[i]) * tail_size;
    if (section_size < smallest_section_size)
      smallest_section_size = section_size;
  }
  
  MSHADOW_TYPE_SWITCH(input_data.type_flag_, DType, {
    size_t global_section_size = smallest_section_size;
    if (leading_size * original_n_sections < 512) {
      // force smaller (limit 128) sections to increase number of blocks
      global_section_size = std::min<size_t>(128, global_section_size);
    }
    size_t block_size = size_t((smallest_section_size + 32 - 1) / 32) * 32;
    block_size = std::min<size_t>(512, block_size);

    size_t n_sections_block = 1;
    if (global_section_size < size2activate_join_sectors) {
      // compute several sections with same block, iterate over leading dims
      n_sections_block = size_t(size2activate_join_sectors / smallest_section_size); 
      block_size = size2activate_join_sectors;
    }

    // redefine sections to improve parallelism   
    std::vector<DType*> new_outputs;
    std::vector<size_t> out_strides;
    std::vector<size_t> in_strides;
    std::vector<size_t> n_elements;
    std::vector<size_t> original_section_sizes;
    std::vector<size_t> accum_elems_block;

    size_t new_n_sections = 0;
    size_t accum_size_split_dim = 0;
    size_t accum_elems = 0;
    for (int i=0; i < original_n_sections; ++i) {
      size_t this_section_size = (indices[i+1] - indices[i]) * tail_size;
      if (this_section_size > global_section_size) {
        // split farther
        for (int j=0; j<this_section_size; j+=global_section_size) {
          new_outputs.push_back(outputs[i].dptr<DType>());
          out_strides.push_back(j);
          in_strides.push_back(accum_size_split_dim);
          size_t remaining_elemts = this_section_size - j;
          size_t n_elems = std::min(remaining_elemts, global_section_size);
          n_elements.push_back(n_elems);
          original_section_sizes.push_back(this_section_size);
          accum_elems_block.push_back(accum_elems);
          new_n_sections++;
          accum_size_split_dim += n_elems;
          if (new_n_sections % n_sections_block == 0)
            accum_elems = 0;
          else
            accum_elems += n_elems;
        }
      } else {
        new_outputs.push_back(outputs[i].dptr<DType>());
        out_strides.push_back(0);
        in_strides.push_back(accum_size_split_dim);
        n_elements.push_back(this_section_size);
        original_section_sizes.push_back(this_section_size);
        accum_elems_block.push_back(accum_elems);
        new_n_sections++;
        accum_size_split_dim += this_section_size;
        if (new_n_sections % n_sections_block == 0)
            accum_elems = 0;
        else
            accum_elems += this_section_size;
      }
    }
    n_sections_block = std::min<size_t>(n_sections_block, new_n_sections);
    size_t n_iters_leading_dims = n_sections_block;

    for(int processed_sections=0; processed_sections<new_n_sections; processed_sections+=split_max_sections) {    
      size_t remaining_sections = new_n_sections - processed_sections;
      // set parameters
      split_tensor_data<DType> params{};
      params.num_sections = std::min(remaining_sections, split_max_sections);
      params.inputs[0] = input_data.dptr<DType>();
      for (int i=0; i < params.num_sections; i++) {
        params.outputs[i] = new_outputs[processed_sections + i];
        params.in_strides[i] = in_strides[processed_sections + i];
        params.out_strides[i] = out_strides[processed_sections + i];
        params.n_elements[i] = n_elements[processed_sections + i];
        params.original_section_sizes[i] = original_section_sizes[processed_sections + i];
        params.accum_elems_block[i] = accum_elems_block[processed_sections + i];
      }
      // load type: we need to check that all sector sizes are multiple of ltype 
      int ltype = get_load_type_split<DType>(params);
      MXNET_LOAD_TYPE_SWITCH(ltype, LType, {
        CHECK_LE(sizeof(DType), sizeof(LType));
        size_t blocks_split_dim = (params.num_sections + n_sections_block - 1) / n_sections_block;
        size_t blocks_leading_dim = (leading_size + n_iters_leading_dims - 1) / n_iters_leading_dims;
        size_t n_blocks =  blocks_leading_dim * blocks_split_dim;
        size_t amount_shared_mem = 0;
        if (n_sections_block > 1) {
          amount_shared_mem = n_sections_block * size2activate_join_sectors * sizeof(DType);
          printf("V0 Launching %lu blocks with %lu threads Shared %lu global_section_size %lu\n", n_blocks, block_size, amount_shared_mem, global_section_size);
          split_tensor_kernel<true, LType><<<n_blocks, block_size, amount_shared_mem, s->stream_>>>
                (params, accum_size_split_dim, blocks_split_dim, n_sections_block,
                 leading_size, n_iters_leading_dims);
        } else {
          printf("V1 Launching %lu blocks with %lu threads Shared %lu global_section_size\n", n_blocks, block_size, amount_shared_mem, global_section_size);
          split_tensor_kernel<false, LType><<<n_blocks, block_size, amount_shared_mem, s->stream_>>>
                (params, accum_size_split_dim, blocks_split_dim, n_sections_block,
                 leading_size, n_iters_leading_dims);
        }
      });
    }   
  }); 
}

NNVM_REGISTER_OP(Reshape)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(Flatten)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(transpose)
.set_attr<FCompute>("FCompute<gpu>", Transpose<gpu>);

NNVM_REGISTER_OP(expand_dims)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(slice)
.set_attr<FCompute>("FCompute<gpu>", SliceOpForward<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SliceEx<gpu>);

NNVM_REGISTER_OP(_backward_slice)
.set_attr<FCompute>("FCompute<gpu>", SliceOpBackward<gpu>);

NNVM_REGISTER_OP(_slice_assign)
.set_attr<FCompute>("FCompute<gpu>", SliceAssignOpForward<gpu>);

NNVM_REGISTER_OP(_slice_assign_scalar)
.set_attr<FCompute>("FCompute<gpu>", SliceAssignScalarOpForward<gpu>);

NNVM_REGISTER_OP(slice_axis)
.set_attr<FCompute>("FCompute<gpu>", SliceAxis<gpu>);

NNVM_REGISTER_OP(_backward_slice_axis)
.set_attr<FCompute>("FCompute<gpu>", SliceAxisGrad_<gpu>);

NNVM_REGISTER_OP(slice_like)
.set_attr<FCompute>("FCompute<gpu>", SliceLikeForward<gpu>);

NNVM_REGISTER_OP(_backward_slice_like)
.set_attr<FCompute>("FCompute<gpu>", SliceLikeBackward<gpu>);

NNVM_REGISTER_OP(clip)
.set_attr<FCompute>("FCompute<gpu>", Clip<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ClipEx<gpu>);

NNVM_REGISTER_OP(_backward_clip)
.set_attr<FCompute>("FCompute<gpu>", ClipGrad_<gpu>);

NNVM_REGISTER_OP(repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpForward<gpu>);

NNVM_REGISTER_OP(_backward_repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpBackward<gpu>);

NNVM_REGISTER_OP(tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpForward<gpu>);

NNVM_REGISTER_OP(_backward_tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpBackward<gpu>);

NNVM_REGISTER_OP(reverse)
.set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(_backward_reverse)
.set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpForward<gpu>);

NNVM_REGISTER_OP(_backward_stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpBackward<gpu>);

NNVM_REGISTER_OP(squeeze)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(_backward_squeeze)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(depth_to_space)
.set_attr<FCompute>("FCompute<gpu>", DepthToSpaceOpForward<gpu>);

NNVM_REGISTER_OP(space_to_depth)
.set_attr<FCompute>("FCompute<gpu>", SpaceToDepthOpForward<gpu>);

NNVM_REGISTER_OP(_split_v2)
.set_attr<FCompute>("FCompute<gpu>", SplitOpForward<gpu>);

NNVM_REGISTER_OP(_split_v2_backward)
.set_attr<FCompute>("FCompute<gpu>", SplitOpBackward<gpu>);

}  // namespace op
}  // namespace mxnet
