#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "powermode_export.h"
// #include "arm_nn_compiler.h"
#include "arm_nnfunctions.h"
// #include "arm_nnsupportfunctions.h"

#define WE2_CHIP_VERSION_C		0x8538000c
#define FRAME_CHECK_DEBUG		1
#ifdef TRUSTZONE_SEC
#ifdef FREERTOS
/* Trustzone config. */
//
/* FreeRTOS includes. */
//#include "secure_port_macros.h"
#else
#if (__ARM_FEATURE_CMSE & 1) == 0
#error "Need ARMv8-M security extensions"
#elif (__ARM_FEATURE_CMSE & 2) == 0
#error "Compile with --cmse"
#endif
#include "arm_cmse.h"
//#include "veneer_table.h"
//
#endif
#endif

#include "WE2_device.h"
#include "spi_master_protocol.h"
#include "hx_drv_spi.h"
#include "spi_eeprom_comm.h"
#include "board.h"
#include "xprintf.h"
#include "allon_sensor_tflm.h"
#include "board.h"
#include "WE2_core.h"
#include "hx_drv_scu.h"
#include "hx_drv_swreg_aon.h"
#ifdef IP_sensorctrl
#include "hx_drv_sensorctrl.h"
#endif
#ifdef IP_xdma
#include "hx_drv_xdma.h"
#include "sensor_dp_lib.h"
#endif
#ifdef IP_cdm
#include "hx_drv_cdm.h"
#endif
#ifdef IP_gpio
#include "hx_drv_gpio.h"
#endif
#include "hx_drv_pmu_export.h"
#include "hx_drv_pmu.h"
#include "powermode.h"
//#include "dp_task.h"
#include "BITOPS.h"

#include "cisdp_sensor.h"
#include "event_handler.h"
#include "common_config.h"
#include "person_detect_model_data.h"

#ifdef EPII_FPGA
#define DBG_APP_LOG             (1)
#else
#define DBG_APP_LOG             (1)
#endif
#if DBG_APP_LOG
    #define dbg_app_log(fmt, ...)       xprintf(fmt, ##__VA_ARGS__)
#else
    #define dbg_app_log(fmt, ...)
#endif

#define MAX_STRING  100
#define DEBUG_SPIMST_SENDPICS		(0x01) //0x00: off/ 0x01: JPEG/0x02: YUV422/0x03: YUV420/0x04: YUV400/0x05: RGB
#define SPI_SEN_PIC_CLK				(10000000)
#define DWT_CYCCNT_START() DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk
#define DWT_CYCCNT_STOP() DWT->CTRL &= ~DWT_CTRL_CYCCNTENA_Msk
#define DWT_CYCCNT_RESET() DWT->CYCCNT = 0
#define DWT_CYCCNT_GET() (DWT->CYCCNT)
#define DWT_CYCCNT_EN() CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk

void initialize_BMM_metadata(cmsis_nn_fc_params *fc_params, cmsis_nn_per_tensor_quant_params *quant_params, cmsis_nn_dims *input_dims, cmsis_nn_dims *filter_dims, cmsis_nn_dims *output_dims) {
    fc_params->input_offset = 0;
    fc_params->filter_offset = 0;
    fc_params->output_offset = 0;
    fc_params->activation.min = INT8_MIN;
    fc_params->activation.max = INT8_MAX;

    quant_params->multiplier = 1113206695;
    quant_params->shift = -4;

    input_dims->n = LHS_ROW; 
    input_dims->h = 1;
    input_dims->w = 1;
    input_dims->c = LHS_COL;

    filter_dims->n = RHS_ROW;
    filter_dims->h = 1;
    filter_dims->w = 1;
    filter_dims->c = RHS_COL;

    output_dims->n = LHS_ROW;
    output_dims->h = 1;
    output_dims->w = 1;
    output_dims->c = RHS_COL;
    return;
}

void BMM_Csr_test_lr(const int8_t *csr_data, const int32_t *csr_indices, const int32_t *csr_ptr, int8_t *input, int8_t *output) {
    cmsis_nn_context *ctx = NULL;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, output_dims;
    const int32_t *bias = NULL;
    const cmsis_nn_dims *bias_dims = NULL;
    initialize_BMM_metadata(&fc_params, &quant_params, &input_dims, &filter_dims, &output_dims);
    DWT_CYCCNT_START();
    DWT_CYCCNT_RESET();

    arm_csr_s8_lr(
        ctx, &fc_params, &quant_params, &input_dims,
        csr_data, csr_indices, csr_ptr,
        &filter_dims, input,
        bias_dims, bias,
        &output_dims, output
    );

    uint32_t cyccnt = DWT_CYCCNT_GET();
    DWT_CYCCNT_STOP();
    printf("bmm(Csr lr) Cycle Count: %ld\n", cyccnt);
	return;
}

void BMM_Rosko(const int8_t *A_p, const int32_t *loc_m, const int32_t *col_idx_rosko, const int32_t *nnz, int8_t *input, int8_t *output) {
	cmsis_nn_context *ctx = NULL;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, output_dims;
    const int32_t *bias = NULL;
    const cmsis_nn_dims *bias_dims = NULL;
    initialize_BMM_metadata(&fc_params, &quant_params, &input_dims, &filter_dims, &output_dims);
    DWT_CYCCNT_START();
    DWT_CYCCNT_RESET();

    arm_rosko(
        ctx, &fc_params, &quant_params, &input_dims,
        A_p, loc_m, col_idx_rosko, nnz,
        &filter_dims, input,
        &output_dims, output
    );

    uint32_t cyccnt = DWT_CYCCNT_GET();
    DWT_CYCCNT_STOP();
    printf("bmm(Rosko) Cycle Count: %ld\n", cyccnt);
	return;
}


void BMM_Fourrows_test_consec(const int8_t *nz_val, const int32_t *col_idx, const int32_t *start_idx, int8_t *input, int8_t *output) {
	cmsis_nn_context *ctx = NULL;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, output_dims;
    const int32_t *bias = NULL;
    const cmsis_nn_dims *bias_dims = NULL;
    initialize_BMM_metadata(&fc_params, &quant_params, &input_dims, &filter_dims, &output_dims);
    DWT_CYCCNT_START();
    DWT_CYCCNT_RESET();

    arm_fourrows_s8_consecutive(
        ctx, &fc_params, &quant_params, &input_dims,
        nz_val, col_idx, start_idx,
        &filter_dims, input,
        &output_dims, output
    );

    uint32_t cyccnt = DWT_CYCCNT_GET();
    DWT_CYCCNT_STOP();
    printf("bmm(Fourrows Consecutive) Cycle Count: %ld\n", cyccnt);
	return;
}

void BMM_Fourrows_test_tiling(const int8_t *nz_val, const int32_t *col_idx, const int32_t *start_idx, int8_t *input, int8_t *output) {
	cmsis_nn_context *ctx = NULL;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, output_dims;
    const int32_t *bias = NULL;
    const cmsis_nn_dims *bias_dims = NULL;
    initialize_BMM_metadata(&fc_params, &quant_params, &input_dims, &filter_dims, &output_dims);
    DWT_CYCCNT_START();
    DWT_CYCCNT_RESET();

    arm_fourrows_s8_tiling(
        ctx, &fc_params, &quant_params, &input_dims,
        nz_val, col_idx, start_idx,
        &filter_dims, input,
        &output_dims, output
    );

    uint32_t cyccnt = DWT_CYCCNT_GET();
    DWT_CYCCNT_STOP();
    printf("bmm(Fourrows Tiling) Cycle Count: %ld\n", cyccnt);
	return;
}

void BMM_FC_test(const int8_t *adj_mat, int8_t *input, int8_t *output) {
	cmsis_nn_context *ctx = (cmsis_nn_context *)malloc(sizeof(cmsis_nn_context));
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, output_dims;
    const int32_t *bias = NULL;
    const cmsis_nn_dims *bias_dims = NULL;
    initialize_BMM_metadata(&fc_params, &quant_params, &input_dims, &filter_dims, &output_dims);

    int32_t buffer[RHS_COL] = {0};
    ctx->buf = (void *)buffer;

    DWT_CYCCNT_START();
    DWT_CYCCNT_RESET();
 	arm_fully_connected_s8(
        ctx, &fc_params, &quant_params, &input_dims,
        adj_mat, &filter_dims,
        input, bias_dims, bias,
        &output_dims, output
	);
    uint32_t cyccnt = DWT_CYCCNT_GET();
    DWT_CYCCNT_STOP();
    printf("bmm(FC) Cycle Count: %ld\n", cyccnt);
	return;
}

void print_output(int8_t *output, char *msg) {
    puts(msg);
    for(int i = 0; i < LHS_ROW; i++) {
        for(int j = 0; j < RHS_COL; j++) {
            printf("%d, ", output[i * RHS_COL + j]);
        }
    }
    return;
}


/*******************************************************************************
 * Code
 ******************************************************************************/
/*!
 * @brief Main function
 */
int app_main(void) {
    DWT_CYCCNT_EN();
    static int8_t input[RHS_ROW * RHS_COL] __attribute__((section(".arr"), aligned(4))) = {0};
	static int8_t input_T[RHS_COL * RHS_ROW] __attribute__((section(".arr"), aligned(4))) = {0};
    static int8_t output[LHS_ROW * RHS_COL] __attribute__((section(".arr"), aligned(4))) = {0};

    int im2col_cnt = 0;
    for(int i = input_h_start; i < input_h_end; i += stride_h) {
        int slide_h = i + filter_h;
        if(slide_h > input_h_end) {            
            break;
        }
        for(int j = input_w_start; j < input_w_end; j += stride_w) {
            int slide_w = j + filter_w;
            if(slide_w > input_w_end) {                
                break;
            }
            // c -> w -> h
            int8_t *input_buf = input + im2col_cnt;
            for(int h = i; h < slide_h; h++) {
                for(int w = j; w < slide_w; w++) {
                    for(int c = 0; c < input_c; c++) {
                        int idx = c + w * input_c + h * input_w_end * input_c;
                        *input_buf = Input[idx];
                        input_buf += RHS_COL;
                    }
                }
            }
            im2col_cnt++;
        }
    }
    printf("im2colcnt: %d\n", im2col_cnt);

    int idx = 0;
    for(int i = 0; i < RHS_COL; i++) {
        for(int j = 0; j < RHS_ROW; j++) {
            input_T[idx++] = input[i + j * RHS_COL];
        }
    }
	
    BMM_FC_test(adj_mx, input_T, output); 
    // print_output(output, "FC result:\n");

    memset(output, 0, sizeof(output));
    BMM_Csr_test_lr(csr_data, csr_indices, csr_ptr, input, output); 
    // print_output(output, "Csr lr result:\n");

    // memset(output, 0, sizeof(output));
    // BMM_Fourrows_test_tiling(nz_val, col_idx, start_idx, input, output);
    // print_output(output, "Fourrows Tiling result:\n");

    memset(output, 0, sizeof(output)); 
    BMM_Fourrows_test_consec(nz_val, col_idx, start_idx, input, output);
    // print_output(output, "Fourrows consecutive result:\n"); 
    
    memset(output, 0, sizeof(output)); 
    BMM_Rosko(A_p, loc_m, col_idx_rosko, nnz, input, output);
    // print_output(output, "Rosko result:\n"); 

    /* GNN FC & FIR */
    // static int8_t input[RHS_ROW * RHS_COL] __attribute__((section(".arr"), aligned(4))) = {0};
    // static int8_t input_T[RHS_COL * RHS_ROW] __attribute__((section(".arr"), aligned(4))) = {0};
    // static int8_t output[LHS_ROW * RHS_COL] __attribute__((section(".arr"), aligned(4))) = {0};

    // memcpy(input, Input, RHS_ROW * RHS_COL);
    // int idx = 0;
    // for(int i = 0; i < RHS_COL; i++) {
    //     for(int j = 0; j < RHS_ROW; j++) {
    //         input_T[idx++] = input[i + j * RHS_COL];
    //     }
    // }
    // BMM_FC_test(adj_mx, input_T, output); 
    // print_output(output, "FC result:\n"); 

    // memset(output, 0, sizeof(output));
    // BMM_Csr_test_lr(csr_data, csr_indices, csr_ptr, input, output); 
    // print_output(output, "Csr lr result:\n");

    // memset(output, 0, sizeof(output));
    // BMM_Fourrows_test_tiling(nz_val, col_idx, start_idx, input, output); 
    // print_output(output, "Fourrows Tiling result:\n");

    // memset(output, 0, sizeof(output));
    // BMM_Fourrows_test_consec(nz_val, col_idx, start_idx, input, output); 
    // print_output(output, "Fourrows consecutive result:\n");
	return 0;
}
