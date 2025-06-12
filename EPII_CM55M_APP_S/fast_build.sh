#!/bin/bash
# we2_build.sh
# 簡易範例：移動目錄、複製檔案、並執行產生工具

make clean || {
  echo "make clean 失敗"
  exit 1
}

# 0.1 make -j8
make -j8 || {
  echo "make -j8 失敗"
  exit 1
}

# 1. 進入 we2_image_gen_local 目錄
cd ../we2_image_gen_local/ || {
  echo "切換目錄失敗，請確認路徑是否正確"
  exit 1
}

# 2. 複製 EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf 到 input_case1_secboot/
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf input_case1_secboot/ || {
  echo "複製檔案失敗，請確認路徑及檔名是否正確"
  exit 1
}

# 3. 執行 we2_local_image_gen
./we2_local_image_gen project_case1_blp_wlcsp.json || {
  echo "we2_local_image_gen 執行失敗"
  exit 1
}

cd .. || {
  echo "切換目錄失敗"
  exit 1
}

echo "=== Start sending img through serial ==="

python xmodem/xmodem_send.py --port=/dev/ttyACM0 --baudrate=921600 --protocol=xmodem --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img || {
  echo "send img failed"
  exit 1
}

