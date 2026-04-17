# IMU BI Preprocessing

This folder contains a standalone utility for converting bilateral OpenSim IMU outputs into filtered 9-IMU CSV files.

The output format is:

```text
time
pelvis_imu_acc_x/y/z
tibia_r_imu_acc_x/y/z
femur_r_imu_acc_x/y/z
tibia_l_imu_acc_x/y/z
femur_l_imu_acc_x/y/z
calcn_r_imu_acc_x/y/z
calcn_l_imu_acc_x/y/z
thigh_r_imu_acc_x/y/z
thigh_l_imu_acc_x/y/z
pelvis_imu_gyr_x/y/z
...
thigh_l_imu_gyr_x/y/z
```

That is 9 IMUs x 6 channels = 54 IMU channels, plus `time`.

## Process Existing IMU_BI_CSV Files

```bash
python imu_processing/process_imu_bi_to_9imu_csv.py \
  --input IMU_Data_Process/AB03_Amy/IMU_BI_CSV \
  --output IMU_Data_Process/AB03_Amy/IMU_BI_CSV_filtered \
  --source csv \
  --cutoff-hz 15
```

## Process OpenSim IMU_BI_Data .sto Files

```bash
python imu_processing/process_imu_bi_to_9imu_csv.py \
  --input IMU_Data_Process/AB02_Rajiv/LG_Exo/IMU_BI_Data \
  --output IMU_Data_Process/AB02_Rajiv/LG_Exo/IMU_BI_CSV_filtered \
  --source sto \
  --cutoff-hz 15
```

Each trial folder is expected to contain one linear acceleration `.sto` file and one angular velocity `.sto` file.

## Notes

- The default low-pass filter is a zero-phase 4th-order Butterworth filter at 15 Hz.
- Sampling frequency is inferred from the `time` column unless `--fs-hz` is provided.
- Accelerometer units are kept as-is by default. Use `--acc-unit-scale auto` to multiply acceleration by 1000 when the median acceleration magnitude suggests the data are in pipeline units rather than m/s^2.
- Use `--trial-regex` to process only selected trials.
