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
 * \file np_einsum_op.cu
 * \brief GPU Implementation of numpy-compatible einsum
 */

#include "./np_einsum_op-inl.h"
#include <nvToolsExt.h>

namespace mxnet {
namespace op {

inline void ComputePairOperands(const TBlob& left,
                                const TBlob& right,
                                int* op_transpose,
                                std::vector<std::vector<std::vector<int>>>& indices,
                                int* need_permutation,
                                std::vector<TShape>& permutations,
                                int* lead_dims,
                                int* strides,
                                mshadow::Tensor<gpu, 1, char> &workspace,
                                TBlob output,
                                const OpContext& ctx) {

  mxnet_op::Stream<gpu>* stream = ctx.get_stream<gpu>();
  CHECK(left.ndim()==right.ndim()) << "number of dimensions must match";

  if (indices[0][1].size() == 0) {
    // not channel_indices

    // start summing axis where required
    // permutations as output
    // add missing dimmensions
    // bcast_like
    // elementwise_mul

  } else {
    // solved by GEMM
    float* a = right.FlatTo2D<gpu, float>(stream).dptr_;
    float* b = left.FlatTo2D<gpu, float>(stream).dptr_;
    float* c = output.FlatTo2D<gpu, float>(stream).dptr_;

    // start summing/reduce axis where required
    // https://github.com/apache/incubator-mxnet/blob/638622f37dcc4ef4b36dcabfd3d7a695fdb7d4c9/src/operator/tensor/broadcast_reduce_op.h#L687

    // perform any transformation if required before GEMM
    size_t pos_wspace = 0;
    if (need_permutation[1]){
      // create new TBlob with the future shape after permutation
      TShape shape_tblob(permutations[1].ndim(), 1);
      for (int i=0; i<permutations[1].ndim(); ++i) {
        shape_tblob[i] = right.size(permutations[1][i]);
      }
      Tensor<gpu, 1, float> aux(reinterpret_cast<float*>(&workspace[pos_wspace]),  //fixed dtype
        Shape1(right.Size()), stream);
      TBlob tranposed(aux);
      tranposed = tranposed.reshape(shape_tblob);
      pos_wspace += tranposed.Size() * sizeof(float);  // fixed dtype
      TransposeImpl<gpu, false>(ctx.run_ctx, right, tranposed, permutations[1]);
      a = tranposed.FlatTo2D<gpu, float>(stream).dptr_;
      //printf("a permuted: ");
      //for(int i=0; i<permutations[1].ndim(); i++) printf("%i ",permutations[1][i]);
      //printf("\n");
    }
    if (need_permutation[0]){
      // create new TBlob with the future shape after permutation
      TShape shape_tblob(permutations[0].ndim(), 1);
      for (int i=0; i<permutations[0].ndim(); ++i) {
        shape_tblob[i] = right.size(permutations[0][i]);
      }
      Tensor<gpu, 1, float> aux(reinterpret_cast<float*>(&workspace[pos_wspace]),  //fixed dtype
        Shape1(right.Size()), stream);
      TBlob tranposed(aux);
      tranposed = tranposed.reshape(shape_tblob);
      pos_wspace += tranposed.Size() * sizeof(float);  // fixed dtype
      TransposeImpl<gpu, false>(ctx.run_ctx, left, tranposed, permutations[0]);
      b = tranposed.FlatTo2D<gpu, float>(stream).dptr_;
      //printf("b permuted: ");
      //for(int i=0; i<permutations[0].ndim(); i++) printf("%i ",permutations[0][i]);
      //printf("\n");
    }
    // if output requires permutation, set teporal shape here
    TBlob* tranposed_output;
    if (need_permutation[2]){
      TShape shape_tblob(permutations[2].ndim(), 1);
      for (int i=0; i<permutations[2].ndim(); ++i) {
        shape_tblob[i] = output.size(permutations[2][i]);
      }
      Tensor<gpu, 1, float> aux(reinterpret_cast<float*>(&workspace[pos_wspace]),  //fixed dtype
        Shape1(output.Size()), stream);
      TBlob tranposed(aux);
      tranposed = tranposed.reshape(shape_tblob);
      c = tranposed.FlatTo2D<gpu, float>(stream).dptr_;
      tranposed_output = &tranposed;
    }

    // GEMM
    int m = 1;
    int n = 1;
    int k = 1;
    int n_batches = 1;
    //  nbatches: batched dims from left
    for (int i=0; i<indices[0][0].size(); i++) {
      n_batches *= left.size(indices[0][0][i]);
      // check right the same
    }
    // k: channel_indices from left
    for (int i=0; i<indices[0][1].size(); i++) {
      k *= left.size(indices[0][1][i]);
      // check right the same
    }
    // m: independent_indices from right
    for (int i=0; i<indices[1][4].size(); i++) {
      m *= right.size(indices[1][4][i]);
    }
    // n: independent_indices from left
    for (int i=0; i<indices[0][3].size(); i++) {
      n *= left.size(indices[0][3][i]);
    }
    int ld_a = lead_dims[1];
    int ld_b = lead_dims[0];
    int ld_c = lead_dims[2];
    int stride_a = strides[1];
    int stride_b = strides[0];
    int stride_c = strides[2];

    cublasOperation_t operation_a = CUBLAS_OP_T;
    cublasOperation_t operation_b = CUBLAS_OP_N;
    if(op_transpose[1])
      operation_a = CUBLAS_OP_N;
    if(op_transpose[0])
      operation_b = CUBLAS_OP_T;

    using namespace mxnet::common::cuda;
    CHECK_EQ(stream->blas_handle_ownership_, mshadow::Stream<gpu>::OwnHandle)
      << "Must init CuBLAS handle in stream";

    cublasHandle_t blas_handle = mshadow::Stream<gpu>::GetBlasHandle(stream);
    auto err = CUBLAS_STATUS_SUCCESS;

    float alpha = 1.0f;
    float beta = 0.0f;

    //printf("m %i n %i k %i\n",m, n, k);
    //printf("lda %i ldb %i ldc %i\n", ld_a, ld_b, ld_c);
    //printf("strideA %i strideB %i strideC %i, NBATCHES %i\n", stride_a, stride_b, stride_c, n_batches);
    //printf("op_transposeA %i op_transposeB %i\n", op_transpose[1], op_transpose[0]);

    err = cublasGemmStridedBatchedEx(
      blas_handle, operation_a, operation_b,
      m, n, k,
      &alpha,
      a, CUDA_R_32F, ld_a, stride_a, // lead dims ???
      b, CUDA_R_32F, ld_b, stride_b, // lead dims ???
      &beta,
      c, CUDA_R_32F, ld_c, stride_c, // lead dims ???
      n_batches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "cublasGemmStridedBatchedEx fail.";

    // if output requires permutation, permute temp into output
    if (need_permutation[2]){
      TransposeImpl<gpu, false>(ctx.run_ctx, *tranposed_output, output, permutations[2]);
    }
  }
}

template<bool back>
inline void NumpyEinsumProcesGPU(const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs,
                                 const char *equation,
                                 const int n_operands,
                                 const OpContext& ctx) {
  //printf("EQUATION: %s\n", equation);
  using namespace mxnet_op;
  mxnet_op::Stream<gpu>* stream = ctx.get_stream<gpu>();

  cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(stream));
  nvtxRangePush("INIT");

  CHECK(n_operands <= 2) << "too many operands provided to einstein sum function";
  CHECK(n_operands >= 1) << "not enough operands provided to einstein sum function";

  //MXNET_ACC_TYPE_SWITCH(out_data.type_flag_, DType, AType, {
  //})

  constexpr size_t n_letters = 26*2; // lower & upper case
  int n_indices= 0;
  std::vector<std::vector<int>> operands_indices; // indices ids for each operand
  int letter_to_index[n_letters]; // to get id from a letter
  int letter_occurrences[n_letters];
  std::vector<int> indices_op_ocurrences; // 0 or 1: once, 2: twice
  std::fill_n(letter_to_index, n_letters, -1);
  std::fill_n(letter_occurrences, n_letters, 0);

  // 1. Parse each operand
  for (size_t op_id = 0; op_id < 2; ++op_id) {
    int n_dims_op = 0;
    std::vector<int> op_indices; // id indices for this op
    size_t op_length = static_cast<int>(strcspn(equation, ",-"));

    CHECK(!(op_id == n_operands-1 && equation[op_length] == ','))
      << "more operands provided to einstein sum function "
      << "than specified in the subscripts string";
    CHECK(!(op_id < n_operands-1 && equation[op_length] != ','))
      << "fewer operands provided to einstein sum function "
      << "than specified in the subscripts string";

    for (size_t n = 0; n < op_length; ++n) {
      auto c = equation[n];
      bool contains_ellipsis = false;
      if (c == '.') { // ellipsis
        CHECK(!(contains_ellipsis != 0 || n + 2 >= op_length
                || equation[++n] != '.' || equation[++n] != '.'))
            << "einstein sum subscripts string contains a "
            << "'.' that is not part of an ellipsis ('...') "
            << "in operand number"
            << op_id;
        contains_ellipsis = true;

      } else if (c < 'A' || c > 'z' || !isalpha(c)){
        CHECK(c == ' ')
            << "invalid subscript '" << static_cast<char>(c)
            << "' in einstein sum "
            << "subscripts string, subscripts must "
            << "be letters";
      } else {
        int letter_num = c-'A';
        if (letter_to_index[letter_num] == -1) {
          // new index
          letter_to_index[letter_num] = n_indices;
          n_indices++;
          indices_op_ocurrences.push_back(op_id);
        } else {
          if (indices_op_ocurrences[letter_to_index[letter_num]] != op_id)
              indices_op_ocurrences[letter_to_index[letter_num]] = 2;
        }
        letter_occurrences[letter_num]++;
        op_indices.push_back(letter_to_index[letter_num]);
        n_dims_op++;
      }
    }
    //CHECK(dims_in_term == tensors[operand].dim(), "dimension mismatch for operand ", operand, ": equation ", dims_in_term, " tensor ", tensors[operand].dim());
    operands_indices.push_back(std::move(op_indices));

    // Move to the next operand
    equation += op_length;
    if (op_id < n_operands - 1) {
      equation++;
    }
  }

  // 2. Parse or infer output
  std::vector<int> indices_to_output_pos(n_indices, -1);
  std::vector<int> op_indices; // id indices for output
  int n_output_dims = 0;
  if (equation[0] == '\0') {
    // infer output
    // the ellipsis (if in the lhs) comes first
    //if (num_ell_idxes >= 0) {
    //  for (int64_t i = 0; i < num_ell_idxes; ++i) {
    //    indices_to_output_pos[first_ell_idx + i] = num_output_dims;
    //    num_output_dims++;
    //  }
    //}
    // indices that occur exactly once in alphabetic order
    for (size_t l = 0; l < n_letters; ++l) {
      if (letter_occurrences[l] == 1) {
        indices_to_output_pos[letter_to_index[l]] = n_output_dims;
        op_indices.push_back(letter_to_index[l]);
        n_output_dims++;
      }
    }
  } else {
    CHECK(equation[0] == '-' && equation[1] == '>')
      << "einstein sum subscript string does not "
      << "contain proper '->' output specified";
    equation += 2;
    // parse the output
    size_t output_length = strlen(equation);
    for (size_t n = 0; n < output_length; ++n) {
      //printf("out Taking letter n %i\n", n);
      auto c = equation[n];
      bool contains_ellipsis = false;
      if (c == '.') {
        // ellipsis
        CHECK(!(contains_ellipsis != 0 || n + 2 >= output_length
              || equation[++n] != '.' || equation[++n] != '.'))
          << "einstein sum subscripts string contains a "
          << "'.' that is not part of an ellipsis ('...') "
          << "in the output";
        contains_ellipsis = true;
      } else if (c < 'A' || c > 'z' || !isalpha(c)){
        CHECK(c == ' ')
            << "invalid subscript '" << static_cast<char>(c)
            << "' in einstein sum "
            << "subscripts string, subscripts must "
            << "be letters";
      } else {
        int letter_num = c-'A';
        // check that it doesn't occur again
        CHECK(indices_to_output_pos[letter_to_index[letter_num]] == -1)
          << "einstein sum subscripts string includes "
          << "output subscript '" << static_cast<char>(c)
          << "' multiple times";
        // check that it was used in the inputs
        CHECK(letter_occurrences[letter_num] > 0)
          << "einstein sum subscripts string included "
          << "output subscript '" << static_cast<char>(c)
          << "' which never appeared "
          << "in an input";
        indices_to_output_pos[letter_to_index[letter_num]] = n_output_dims;
        op_indices.push_back(letter_to_index[letter_num]);
        n_output_dims++;
      }
    }
  }
  operands_indices.push_back(std::move(op_indices));
  CHECK_GE(n_output_dims, 0);

  // check what dimmensions are summed
  std::vector<std::vector<std::vector<int>>> ops_indices_by_type;
  // vector for each op, with 5 vectors (for each type of dimension):
  // 0: batched, 1: summed channel (GEMM), 2: summed independently
  // 3: independent dim left op, 4: independent dim right op
  // with the list of index positions within the operand
  std::vector<std::vector<int>> type_order;
  // keep record which kind of index is first to calculate permutations

  for (size_t op_id = 0; op_id < 3; op_id++) {
    // for left; right; output
    std::vector<std::vector<int>> indices_by_type;
    std::vector<int> batched_ind;
    std::vector<int> channel_ind;
    std::vector<int> summed_ind;
    std::vector<int> independent_left_ind;
    std::vector<int> independent_right_ind;
    std::vector<int> types;

    for (size_t i = 0; i < operands_indices[op_id].size(); i++) {
      int op_index = operands_indices[op_id][i];
      int dim_out = indices_to_output_pos[op_index];
      if (dim_out == -1) {
        // ommited in output -> summed
        if (indices_op_ocurrences[op_index] == 2) {
          // channel for dot or inner product
          types.push_back(1);
          channel_ind.push_back(i);
        } else {
          // summed independently
          types.push_back(2);
          summed_ind.push_back(i);
        }
      } else {
        if(indices_op_ocurrences[op_index] == 2){
          // in both ops - not summed: batched
          types.push_back(0);
          batched_ind.push_back(i);
        } else {
          // independent not summed
          if (op_id == 0) {
              types.push_back(3);
              independent_left_ind.push_back(i);
          } else if (op_id == 1) {
              types.push_back(4);
              independent_right_ind.push_back(i);
          } else {
            std::vector<int>::iterator it;
            it = find (operands_indices[0].begin(),
                       operands_indices[0].end(),
                       op_index);
            if (it != operands_indices[0].end()) {
              types.push_back(3);
              independent_left_ind.push_back(i);
            } else {
              types.push_back(4);
              independent_right_ind.push_back(i);
            }
          }
        }
      }
    }
    indices_by_type.push_back(batched_ind);
    indices_by_type.push_back(channel_ind);
    indices_by_type.push_back(summed_ind);
    indices_by_type.push_back(independent_left_ind);
    indices_by_type.push_back(independent_right_ind);
    ops_indices_by_type.push_back(indices_by_type);
    type_order.push_back(types);

    /*printf("OP %i \n", op_id);
    for(int j=0;j<channel_ind.size();j++) printf("chanel[%i] %i\n",j,channel_ind[j]);
    for(int j=0;j<summed_ind.size();j++) printf("summed[%i] %i\n",j,summed_ind[j]);
    for(int j=0;j<batched_ind.size();j++) printf("batched_ind[%i] %i\n",j,batched_ind[j]);
    for(int j=0;j<independent_left_ind.size();j++) printf("independent_left_ind[%i] %i\n",j,independent_left_ind[j]);
    for(int j=0;j<independent_right_ind.size();j++) printf("independent_right_ind[%i] %i\n",j,independent_right_ind[j]);*/
  }

  // check what permutations are required:
  // putting togueter dims with same type, solved left to right
  int need_permutation[3]; // left , right, output
  std::fill_n(need_permutation, 3, 0);
  std::vector<TShape> permutations;
  // temporal storage: depends on how many operands need permutations
  size_t workspace_size = 0;
  int lead_dims[3]; // used if batched gemm
  int strides[3]; // used if batched gemm
  int op_transpose[2]; // used if batched gemm, left, right
  std::fill_n(op_transpose, 2, 0);

  if (ops_indices_by_type[0][1].size()){
    // channel indices -> require GEMM
    for (size_t op_id = 0; op_id < 3; op_id++) {
      std::vector<int> type_processed(5, 0);
      int ndims = operands_indices[op_id].size();
      TShape perm(ndims,1);
      int position = 0;
      lead_dims[op_id] = 1;
      strides[op_id] = 1;
      // check if need transpose op
      int first_independent_pos;
      if (op_id < 2){
        first_independent_pos = (op_id==0) ?
            ops_indices_by_type[op_id][3][0] :
            ops_indices_by_type[op_id][4][0];
        if (ops_indices_by_type[op_id][1][0] <
            first_independent_pos) {
          // channel dim before independent dim
          op_transpose[op_id] = 1;
        }
      }
      for (size_t i = 0; i < operands_indices[op_id].size(); i++) {
        int type = type_order[op_id][i];
        if (!type_processed[type]){
          perm[position] = ops_indices_by_type[op_id][type][0];
          position++;
          // update lead_dims & stride
          size_t size_dim;
          if (op_id < 2) size_dim = inputs[op_id].size(i);
          else size_dim = outputs[0].size(i);
          lead_dims[op_id] *= size_dim;
          strides[op_id] *= size_dim;
          // put all dims with same type toguether
          for (size_t d = 1; d < ops_indices_by_type[op_id][type].size(); d++){
            int index_pos = ops_indices_by_type[op_id][type][d];
            if (index_pos != position)
              need_permutation[op_id] = 1;
            perm[position] = index_pos;
            position++;
            // update lead_dims & stride
            size_t size_dim;
            if (op_id < 2) size_dim = inputs[op_id].size(index_pos);
            else size_dim = outputs[0].size(index_pos);
            lead_dims[op_id] *= size_dim;
            strides[op_id] *= size_dim;
          }
          if (type == 0) strides[op_id] = 1; // reset stride if batched
          if (type >=3){
          // reset leading dim if independent dimension and not transpose op
            if (op_id < 2 && !op_transpose[op_id]) lead_dims[op_id] = 1;
            else {
              // excepts if output: only reset if not last independent(l/r)
              if (type == 3 && ops_indices_by_type[op_id][4].size() &&
                  !type_processed[4]) lead_dims[op_id] = 1;
              if (type == 4 && ops_indices_by_type[op_id][3].size() &&
                  !type_processed[3]) lead_dims[op_id] = 1;
            }
          }
          // reset leading dim if channel dimension and transpose op
          if (type == 1 && op_id < 2 && op_transpose[op_id]) lead_dims[op_id] = 1;

          type_processed[type] = 1;
        }
      }
      if (need_permutation[op_id]){
        if (op_id < 2)
          workspace_size += inputs[op_id].Size() * sizeof(float); // float !!
        else
         workspace_size += outputs[0].Size() * sizeof(float); // float !!
      }
      //printf("OP %i lead_dim %i stride %i\n", op_id, lead_dims[op_id], strides[op_id]);
      //printf("Need permutation %i\n", need_permutation[op_id]);
      //for (int i=0; i<ndims;i++) printf("%i ",perm[i]);
      //printf("\n");
      permutations.push_back(perm);
    }
  } else {
    // elementwise mult: permute operands as output
  }

  Tensor<gpu, 1, char> workspace =
    ctx.requested[0].get_space_typed<gpu, 1, char>(Shape1(workspace_size), stream);

  cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(stream));
  nvtxRangePop();
  nvtxRangePush("einsum");

  ComputePairOperands(inputs[0], inputs[1],
                      op_transpose,
                      ops_indices_by_type,
                      need_permutation,
                      permutations,
                      lead_dims,
                      strides,
                      workspace,
                      outputs[0],
                      ctx);

  cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(stream));
  nvtxRangePop();
}

template<typename xpu>
inline void NumpyEinsumForwardGPU(const OpStatePtr& state_ptr,
                                   const OpContext& ctx,
                                   const std::vector<TBlob>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  EinsumOp& state = state_ptr.get_state<EinsumOp>();
  int num_args = state.num_args;
  //int optimize = state.optimize;
  const char* subscripts = state.subscripts.c_str();
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), num_args);
  CHECK_EQ(outputs.size(), 1U);

  NumpyEinsumProcesGPU<0>(inputs, req, outputs, subscripts, num_args, ctx);
}

NNVM_REGISTER_OP(_npi_einsum)
.set_attr<FStatefulCompute>("FStatefulCompute<gpu>", NumpyEinsumForwardGPU<gpu>);
NNVM_REGISTER_OP(_backward_npi_einsum)
.set_attr<FStatefulCompute>("FStatefulCompute<gpu>", NumpyEinsumBackward<gpu>);

}  // namespace op
}  // namespace mxnet
