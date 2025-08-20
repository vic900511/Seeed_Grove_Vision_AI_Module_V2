#ifndef APP_SCENARIO_ALLON_SENSOR_TFLM_PERSON_DETECT_MODEL_DATA_H_
#define APP_SCENARIO_ALLON_SENSOR_TFLM_PERSON_DETECT_MODEL_DATA_H_

#include <stdint.h>

#define LHS_ROW 49
#define LHS_COL 160
#define RHS_ROW 160
#define RHS_COL 960

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

extern const int8_t values[];
extern const uint16_t bitmasks[];
extern const uint8_t bitmaps[];
extern const uint8_t delta_indices[];
extern const int8_t minimums[];
extern const int16_t row_offsets[];
extern const uint32_t nnze;
extern uint8_t idx_buffer[];
extern int16_t group_buffer[];

#endif