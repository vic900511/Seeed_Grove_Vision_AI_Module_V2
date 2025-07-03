/*
 * SPDX-FileCopyrightText: Copyright 2010-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_fully_connected_s8
 * Description:  Fully connected function compatible with TF Lite.
 *
 * $Date:        6 February 2024
 * $Revision:    V.5.3.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup Public
 */

/**
 * @addtogroup FC
 * @{
 */

/*
 * S8 basic fully-connected and matrix multiplication layer function for TensorFlow Lite
 *
 * Refer header file for details.
 *
 */

arm_cmsis_nn_status arm_fully_connected_s8(const cmsis_nn_context *ctx,
                                           const cmsis_nn_fc_params *fc_params,
                                           const cmsis_nn_per_tensor_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int8_t *input,
                                           const cmsis_nn_dims *filter_dims,
                                           const int8_t *kernel,
                                           const cmsis_nn_dims *bias_dims,
                                           const int32_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           int8_t *output)
{
    (void)bias_dims;

    int32_t batch_cnt = input_dims->n;

#if defined(ARM_MATH_MVEI)
    if (ctx->buf == NULL)
    {
        return (ARM_CMSIS_NN_ARG_ERROR);
    }
#endif
    
    const int32_t *kernel_sum = (const int32_t *)ctx->buf;

    while (batch_cnt)
    {
        arm_nn_vec_mat_mult_t_s8(input,
                                 kernel,
                                 kernel_sum,
                                 bias,
                                 output,
                                 fc_params->input_offset,
                                 fc_params->output_offset,
                                 quant_params->multiplier,
                                 quant_params->shift,
                                 filter_dims->n, /* col_dim or accum_depth */   // rhs_cols 9 
                                 output_dims->c, /* row_dim or output_depth */  // rhs_rows 676
                                 fc_params->activation.min,
                                 fc_params->activation.max,
                                 1L,
                                 fc_params->filter_offset);

        input += filter_dims->n;
        output += output_dims->c;
        batch_cnt--;
    }
    return (ARM_CMSIS_NN_SUCCESS);
}


arm_cmsis_nn_status arm_csr_s8_lr(const cmsis_nn_context *ctx,
                                  const cmsis_nn_fc_params *fc_params,
                                  const cmsis_nn_per_tensor_quant_params *quant_params,
                                  const cmsis_nn_dims *input_dims,
                                  const int8_t *csr_data,
                                  const int32_t *csr_indices, 
                                  const int32_t *csr_ptr,
                                  const cmsis_nn_dims *filter_dims,
                                  const int8_t *kernel,
                                  const cmsis_nn_dims *bias_dims,
                                  const int32_t *bias,
                                  const cmsis_nn_dims *output_dims,
                                  int8_t *output) 
{
    int32_t batch_cnt = input_dims->n;
    for(int i = 0; i < batch_cnt; i++) {
        const int32_t col_num = csr_ptr[i + 1] - csr_ptr[i];
        arm_nn_lr_csr_s8_for_bmm(csr_data, csr_indices, col_num, kernel, output, filter_dims->c);
        csr_data += col_num;
        csr_indices += col_num;
        output += output_dims->c;
    }
    return (ARM_CMSIS_NN_SUCCESS);
}

arm_cmsis_nn_status arm_rosko(const cmsis_nn_context *ctx,
                              const cmsis_nn_fc_params *fc_params,
                              const cmsis_nn_per_tensor_quant_params *quant_params,
                              const cmsis_nn_dims *input_dims,
                              const int8_t *A_p,
                              const int32_t *loc_m,
                              const int32_t *col_idx_rosko,
                              const int32_t *nnz,
                              const cmsis_nn_dims *filter_dims, 
                              const int8_t *kernel,
                              const cmsis_nn_dims *output_dims,
                              int8_t *output) 
{
    int32_t lhs_cols = input_dims->c;
    int32_t rhs_cols = filter_dims->c;
    for(int k = 0; k < lhs_cols; k++) {
        if(nnz[k] == 0) {
            break;
        }
        const int8_t *rhs_vec = kernel + rhs_cols * col_idx_rosko[k];
        for(int m = 0; m < nnz[k]; m++) {
            const int8_t lhs_val = *A_p;
            A_p++;
            int8_t *output_start = output + *(loc_m) * rhs_cols;
            loc_m++;
            arm_nn_fourrows_s8_for_bmm(lhs_val, rhs_vec, output_start, rhs_cols);
        }
    }
    return (ARM_CMSIS_NN_SUCCESS);
}

typedef struct { 
    uint8_t num_non_zeros; 
    uint8_t non_zeros[4]; 
} PatternInfo;

static const PatternInfo pattern_lookup_table[15] = {
    {4, {0, 1, 2, 3}}, 
    {3, {0, 1, 2}}, 
    {3, {0, 1, 3}}, 
    {3, {0, 2, 3}}, 
    {3, {1, 2, 3}}, 
    {2, {0, 1}}, 
    {2, {0, 2}}, 
    {2, {0, 3}}, 
    {2, {1, 2}}, 
    {2, {1, 3}}, 
    {2, {2, 3}}, 
    {1, {0}}, 
    {1, {1}},
    {1, {2}},
    {1, {3}}
};


arm_cmsis_nn_status arm_fourrows_s8_consecutive(const cmsis_nn_context *ctx,
                                                const cmsis_nn_fc_params *fc_params,
                                                const cmsis_nn_per_tensor_quant_params *quant_params,
                                                const cmsis_nn_dims *input_dims,
                                                const int8_t *nz_val,
                                                const int32_t *col_idx,
                                                const int32_t *start_idx,
                                                const cmsis_nn_dims *filter_dims, 
                                                const int8_t *kernel,
                                                const cmsis_nn_dims *output_dims,
                                                int8_t *output) 
{
    int nnz_num = 0;
    int end = (input_dims->n + 3) / 4;
    int32_t rhs_cols = filter_dims->c;
    for(int i = 0; i < end; i++) {
        for(uint8_t pattern_id = 0; pattern_id < 15; pattern_id++) {
            const PatternInfo *pattern_info = &pattern_lookup_table[pattern_id];
            const uint8_t num_non_zero = pattern_info->num_non_zeros;
            for(; nnz_num < start_idx[16 * i + pattern_id + 1]; nnz_num += num_non_zero) {
                const int8_t *rhs_vec = kernel + *(col_idx) * rhs_cols;
                col_idx++;
                for(uint8_t n = 0; n < num_non_zero; n++) {
                    int8_t *output_start = output + pattern_info->non_zeros[n] * rhs_cols;
                    arm_nn_fourrows_s8_for_bmm(*(nz_val), rhs_vec, output_start, rhs_cols);
                    nz_val++;
                }
            }
        }
        output += 4 * filter_dims->c;
    }

    return (ARM_CMSIS_NN_SUCCESS);
}

// arm_cmsis_nn_status arm_fourrows_s8_consecutive(const cmsis_nn_context *ctx,
//                                                 const cmsis_nn_fc_params *fc_params,
//                                                 const cmsis_nn_per_tensor_quant_params *quant_params,
//                                                 const cmsis_nn_dims *input_dims,
//                                                 const int8_t *nz_val,
//                                                 const int32_t *col_idx,
//                                                 const int32_t *start_idx,
//                                                 const cmsis_nn_dims *filter_dims, 
//                                                 const int8_t *kernel,
//                                                 const cmsis_nn_dims *output_dims,
//                                                 int8_t *output) 
// {
//     int nnz_num = 0;
//     int end = (input_dims->n + 3) / 4;
//     for(int i = 0; i < end; i++) {
//         int pattern_nnz_num;
//         for(int pattern = 0; pattern < 15; pattern++) {
//             if(pattern == 0) {
//                 pattern_nnz_num = 4;
//             }
//             else if(pattern >= 1 && pattern <= 4) {
//                 pattern_nnz_num = 3;
//             }
//             else if(pattern >= 5 && pattern <= 10) {
//                 pattern_nnz_num = 2;
//             }
//             else {
//                 pattern_nnz_num = 1;
//             }
//             // method 1: 1 lhsval * rhs vec (有 locality)
//             while(nnz_num != start_idx[16 * i + pattern + 1]) {
//                 const int8_t *rhs_vec = kernel + filter_dims->c * (*col_idx);
//                 col_idx++;
//                 int8_t *output_start = output;
//                 nnz_num += pattern_nnz_num;

//                 if(pattern == 0) {
//                     for(int j = 0; j < 4; j++) {
//                         arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                         output_start += filter_dims->c;
//                         nz_val++;
//                     }
//                 }
//                 else if(pattern == 1) {
//                     for(int j = 0; j < 3; j++) {
//                         arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                         output_start += filter_dims->c;
//                         nz_val++;
//                     }
//                 }
//                 else if(pattern == 2) {
//                     for(int j = 0; j < 2; j++) {
//                         arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                         output_start += filter_dims->c;
//                         nz_val++;
//                     }
//                     output_start += filter_dims->c;
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     nz_val++;
//                 }
//                 else if(pattern == 3) {
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     output_start += 2 * filter_dims->c;
//                     nz_val++;
//                     for(int j = 0; j < 2; j++) {
//                         arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                         output_start += filter_dims->c;
//                         nz_val++;
//                     }
//                 }
//                 else if(pattern == 4) {
//                     output_start += filter_dims->c;
//                     for(int j = 0; j < 3; j++) {
//                         arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                         output_start += filter_dims->c;
//                         nz_val++;
//                     }
//                 }
//                 else if(pattern == 5) {
//                     for(int j = 0; j < 2; j++) {
//                         arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                         output_start += filter_dims->c;
//                         nz_val++;
//                     }
//                 }
//                 else if(pattern == 6) {
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     output_start += 2 * filter_dims->c;
//                     nz_val++;
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     nz_val++;
//                 }
//                 else if(pattern == 7) {
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     output_start += 3 * filter_dims->c;
//                     nz_val++;
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     nz_val++;
//                 }
//                 else if(pattern == 8) {
//                     output_start += filter_dims->c;   
//                     for(int j = 0; j < 2; j++) {
//                         arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                         output_start += filter_dims->c;
//                         nz_val++;
//                     }
//                 }
//                 else if(pattern == 9) {
//                     output_start += filter_dims->c;
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     output_start += 2 * filter_dims->c;
//                     nz_val++;
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     nz_val++;
//                 }
//                 else if(pattern == 10) {
//                     output_start += 2 * filter_dims->c;
//                     for(int j = 0; j < 2; j++) {
//                         arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                         output_start += filter_dims->c;
//                         nz_val++;
//                     }
//                 }
//                 else if(pattern == 11) {
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     nz_val++;
//                 }
//                 else if(pattern == 12) {
//                     output_start += filter_dims->c;
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     nz_val++;
//                 }
//                 else if(pattern == 13) {
//                     output_start += 2 * filter_dims->c;
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     nz_val++;
//                 }
//                 else if(pattern == 14) {
//                     output_start += 3 * filter_dims->c;
//                     arm_nn_fourrows_s8_for_bmm(*nz_val, rhs_vec, output_start, filter_dims->c);
//                     nz_val++;
//                 }
//             }
//         }
//         output += 4 * filter_dims->c;
//     }

//     return (ARM_CMSIS_NN_SUCCESS);
// }

arm_cmsis_nn_status arm_fourrows_s8_tiling(const cmsis_nn_context *ctx,
                                           const cmsis_nn_fc_params *fc_params,
                                           const cmsis_nn_per_tensor_quant_params *quant_params,
                                           const cmsis_nn_dims *input_dims,
                                           const int8_t *nz_val,
                                           const int32_t *col_idx,
                                           const int32_t *start_idx,
                                           const cmsis_nn_dims *filter_dims, 
                                           const int8_t *kernel,
                                           const cmsis_nn_dims *output_dims,
                                           int8_t *output) 
{
#if defined(ARM_MATH_MVEI)
    int nnz_num = 0;
    int end = input_dims->n % 4 ? input_dims->n / 4 + 1 : input_dims->n / 4;
    const int32_t col_loop_cnt = (filter_dims->c + 15) / 16;
    for(int i = 0; i < end; i++) {
        int pattern_nnz_num;
        for(int pattern = 0; pattern < 15; pattern++) {
            if(pattern == 0) {
                pattern_nnz_num = 4;
            }
            else if(pattern >= 1 && pattern <= 4) {
                pattern_nnz_num = 3;
            }
            else if(pattern >= 5 && pattern <= 10) {
                pattern_nnz_num = 2;
            }
            else {
                pattern_nnz_num = 1;
            }
            // method 2: 4 lhsval * rhs vec 
            while(nnz_num != start_idx[16 * i + pattern + 1]) {
                const int8_t *rhs_vec = kernel + filter_dims->c * (*col_idx);
                col_idx++;
                int8_t *output_start = output;
                nnz_num += pattern_nnz_num;

                if(pattern == 0) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor[4];
                    lhs_factor[0] = vdupq_n_s8(*nz_val++);
                    lhs_factor[1] = vdupq_n_s8(*nz_val++);
                    lhs_factor[2] = vdupq_n_s8(*nz_val++);
                    lhs_factor[3] = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start;
                        for(int k = 0; k < 4; k++) {
                            int8x16_t product = vmulq_s8(ker_0, lhs_factor[k]);
                            const int8x16_t tmp = vldrbq_z_s8(dst, p);
                            int8x16_t acc = vaddq_s8(tmp, product);
                            vstrbq_p_s8(dst, acc, p);
                            dst += filter_dims->c;
                        }
                        
                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 1) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor[3];
                    lhs_factor[0] = vdupq_n_s8(*nz_val++);
                    lhs_factor[1] = vdupq_n_s8(*nz_val++);
                    lhs_factor[2] = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start;
                        for(int k = 0; k < 3; k++) {
                            int8x16_t product = vmulq_s8(ker_0, lhs_factor[k]);
                            const int8x16_t tmp = vldrbq_z_s8(dst, p);
                            int8x16_t acc = vaddq_s8(tmp, product);
                            vstrbq_p_s8(dst, acc, p);
                            dst += filter_dims->c;
                        }
                        
                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 2) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor[3];
                    lhs_factor[0] = vdupq_n_s8(*nz_val++);
                    lhs_factor[1] = vdupq_n_s8(*nz_val++);
                    lhs_factor[2] = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start;
                        for(int k = 0; k < 3; k++) {
                            if(k == 2) {
                                dst += filter_dims->c;
                            }
                            int8x16_t product = vmulq_s8(ker_0, lhs_factor[k]);
                            const int8x16_t tmp = vldrbq_z_s8(dst, p);
                            int8x16_t acc = vaddq_s8(tmp, product);
                            vstrbq_p_s8(dst, acc, p);
                            dst += filter_dims->c;
                        }

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 3) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor[3];
                    lhs_factor[0] = vdupq_n_s8(*nz_val++);
                    lhs_factor[1] = vdupq_n_s8(*nz_val++);
                    lhs_factor[2] = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start;
                        for(int k = 0; k < 3; k++) {
                            if(k == 1) {
                                dst += filter_dims->c;
                            }
                            int8x16_t product = vmulq_s8(ker_0, lhs_factor[k]);
                            const int8x16_t tmp = vldrbq_z_s8(dst, p);
                            int8x16_t acc = vaddq_s8(tmp, product);
                            vstrbq_p_s8(dst, acc, p);
                            dst += filter_dims->c;
                        }

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 4) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor[3];
                    lhs_factor[0] = vdupq_n_s8(*nz_val++);
                    lhs_factor[1] = vdupq_n_s8(*nz_val++);
                    lhs_factor[2] = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start + filter_dims->c;
                        for(int k = 0; k < 3; k++) {
                            int8x16_t product = vmulq_s8(ker_0, lhs_factor[k]);
                            const int8x16_t tmp = vldrbq_z_s8(dst, p);
                            int8x16_t acc = vaddq_s8(tmp, product);
                            vstrbq_p_s8(dst, acc, p);
                            dst += filter_dims->c;
                        }

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 5) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor[2];
                    lhs_factor[0] = vdupq_n_s8(*nz_val++);
                    lhs_factor[1] = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start;
                        for(int k = 0; k < 2; k++) {
                            int8x16_t product = vmulq_s8(ker_0, lhs_factor[k]);
                            const int8x16_t tmp = vldrbq_z_s8(dst, p);
                            int8x16_t acc = vaddq_s8(tmp, product);
                            vstrbq_p_s8(dst, acc, p);
                            dst += filter_dims->c;
                        }

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 6) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor[2];
                    lhs_factor[0] = vdupq_n_s8(*nz_val++);
                    lhs_factor[1] = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start;
                        int8x16_t product = vmulq_s8(ker_0, lhs_factor[0]);
                        int8x16_t tmp = vldrbq_z_s8(dst, p);
                        int8x16_t acc = vaddq_s8(tmp, product);
                        vstrbq_p_s8(dst, acc, p);

                        dst += 2 * filter_dims->c;
                        product = vmulq_s8(ker_0, lhs_factor[1]);
                        tmp = vldrbq_z_s8(dst, p);
                        acc = vaddq_s8(tmp, product);
                        vstrbq_p_s8(dst, acc, p);

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 7) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor[2];
                    lhs_factor[0] = vdupq_n_s8(*nz_val++);
                    lhs_factor[1] = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start;
                        int8x16_t product = vmulq_s8(ker_0, lhs_factor[0]);
                        int8x16_t tmp = vldrbq_z_s8(dst, p);
                        int8x16_t acc = vaddq_s8(tmp, product);
                        vstrbq_p_s8(dst, acc, p);

                        dst += 3 * filter_dims->c;
                        product = vmulq_s8(ker_0, lhs_factor[1]);
                        tmp = vldrbq_z_s8(dst, p);
                        acc = vaddq_s8(tmp, product);
                        vstrbq_p_s8(dst, acc, p);

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 8) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor[2];
                    lhs_factor[0] = vdupq_n_s8(*nz_val++);
                    lhs_factor[1] = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start + filter_dims->c;
                        for(int k = 0; k < 2; k++) {
                            int8x16_t product = vmulq_s8(ker_0, lhs_factor[k]);
                            const int8x16_t tmp = vldrbq_z_s8(dst, p);
                            int8x16_t acc = vaddq_s8(tmp, product);
                            vstrbq_p_s8(dst, acc, p);
                            dst += filter_dims->c;
                        }

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 9) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor[2];
                    lhs_factor[0] = vdupq_n_s8(*nz_val++);
                    lhs_factor[1] = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start + filter_dims->c;
                        int8x16_t product = vmulq_s8(ker_0, lhs_factor[0]);
                        int8x16_t tmp = vldrbq_z_s8(dst, p);
                        int8x16_t acc = vaddq_s8(tmp, product);
                        vstrbq_p_s8(dst, acc, p);

                        dst += 2 * filter_dims->c;
                        product = vmulq_s8(ker_0, lhs_factor[1]);
                        tmp = vldrbq_z_s8(dst, p);
                        acc = vaddq_s8(tmp, product);
                        vstrbq_p_s8(dst, acc, p);

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 10) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor[2];
                    lhs_factor[0] = vdupq_n_s8(*nz_val++);
                    lhs_factor[1] = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start + 2 * filter_dims->c;
                        for(int k = 0; k < 2; k++) {
                            int8x16_t product = vmulq_s8(ker_0, lhs_factor[k]);
                            const int8x16_t tmp = vldrbq_z_s8(dst, p);
                            int8x16_t acc = vaddq_s8(tmp, product);
                            vstrbq_p_s8(dst, acc, p);
                            dst += filter_dims->c;
                        }

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 11) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start;
                        int8x16_t product = vmulq_s8(ker_0, lhs_factor);
                        const int8x16_t tmp = vldrbq_z_s8(dst, p);
                        int8x16_t acc = vaddq_s8(tmp, product);
                        vstrbq_p_s8(dst, acc, p);                        

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 12) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start + filter_dims->c;
                        int8x16_t product = vmulq_s8(ker_0, lhs_factor);
                        const int8x16_t tmp = vldrbq_z_s8(dst, p);
                        int8x16_t acc = vaddq_s8(tmp, product);
                        vstrbq_p_s8(dst, acc, p);                        

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 13) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start + 2 * filter_dims->c;
                        int8x16_t product = vmulq_s8(ker_0, lhs_factor);
                        const int8x16_t tmp = vldrbq_z_s8(dst, p);
                        int8x16_t acc = vaddq_s8(tmp, product);
                        vstrbq_p_s8(dst, acc, p);                        

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
                else if(pattern == 14) {
                    uint32_t col_cnt = (uint32_t)filter_dims->c;
                    int8x16_t lhs_factor = vdupq_n_s8(*nz_val++);

                    for(int32_t j = 0; j < col_loop_cnt; j++) {
                        mve_pred16_t p = vctp8q(col_cnt);
                        col_cnt -= 16;
                        const int8x16_t ker_0 = vldrbq_z_s8(rhs_vec, p);

                        int8_t *dst = output_start + 3 * filter_dims->c;
                        int8x16_t product = vmulq_s8(ker_0, lhs_factor);
                        const int8x16_t tmp = vldrbq_z_s8(dst, p);
                        int8x16_t acc = vaddq_s8(tmp, product);
                        vstrbq_p_s8(dst, acc, p);                        

                        rhs_vec += 16;
                        output_start += 16;
                    }
                }
            }
        }
        output += 4 * filter_dims->c;
    }
#endif
    return (ARM_CMSIS_NN_SUCCESS);
}

/**
 * @} end of FC group
 */
