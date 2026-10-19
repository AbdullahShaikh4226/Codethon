CLEANED CLOUD RESTORE DATASETS

Files:
- vmware_restore_data_cleaned.csv
- azure_restore_data_cleaned.csv
- aws_restore_data_cleaned.csv
- cloud_restore_combined_harmonized.csv
- restore_dataset_cleaning_report.csv
- restore_time_outlier_review.csv

Cleaning performed:
1. Trimmed whitespace from column names and text values.
2. Normalized repeated whitespace in categorical fields.
3. Converted numeric fields to numeric types and recorded conversion failures.
4. Removed exact duplicate rows.
5. Removed rows with missing required values.
6. Removed rows with non-positive values in required size, restore-time, bandwidth,
   concurrency, or IOPS measurements.
7. Rounded conceptually integral fields to integer types and retained restore time
   to six decimal places.
8. Sorted each provider dataset deterministically by its resource identifier.
9. Flagged restore-time IQR outliers for review, but retained them in the cleaned
   datasets because unusual values can be valid and should not be deleted without
   domain context.

Combined dataset:
The harmonized file contains one row per resource and common fields such as Provider,
Resource_ID, VM_Size_GB, Storage_Type, and Restore_Time_Min. Network bandwidth is
kept with an explicit unit because VMware uses Gbps and Azure uses Mbps. AWS has no
network-bandwidth column in the source. Provider-specific fields remain where relevant;
blank cells mean the field was not provided for that provider, not necessarily zero.

Important:
These steps check data quality and consistency, not whether the records are real,
representative, or suitable for production predictions. Validate source provenance,
units, and outliers before using the data for scientific claims or model evaluation.
