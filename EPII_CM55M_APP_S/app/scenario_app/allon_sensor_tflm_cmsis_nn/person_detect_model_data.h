#ifndef APP_SCENARIO_ALLON_SENSOR_TFLM_PERSON_DETECT_MODEL_DATA_H_
#define APP_SCENARIO_ALLON_SENSOR_TFLM_PERSON_DETECT_MODEL_DATA_H_

#include <stdint.h>

#define LHS_ROW 100
#define LHS_COL 700
#define RHS_ROW 700
#define RHS_COL 142

extern const int input_h_start, input_h_end, input_w_start, input_w_end, input_c;
extern const int filter_h, filter_w;
extern const int stride_h, stride_w;
extern const int8_t Input[];
extern const int8_t adj_mx[];
extern const int8_t nz_val[];
extern const int32_t col_idx[];
extern const int32_t start_idx[];
extern const int8_t csr_data[];
extern const int32_t csr_indices[];
extern const int32_t csr_ptr[];
extern const int8_t A_p[];
extern const int32_t loc_m[];
extern const int32_t col_idx_rosko[];
extern const int32_t nnz[];

#endif