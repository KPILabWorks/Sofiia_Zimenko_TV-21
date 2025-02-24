import pandas as pd
import numpy as np
import time

n_rows = 10_000_000
df = pd.DataFrame({
    'id': np.arange(1, n_rows + 1),
    'value': np.random.rand(n_rows),
    'category': np.random.choice(['A', 'B', 'C', 'D'], size=n_rows),
    'timestamp': pd.date_range(start='2023-01-01', periods=n_rows, freq='min'),
    'integer_value': np.random.randint(1, 100, size=n_rows)
})

def measure_write_time(df, file_format, file_name, write_func, **kwargs):
    start_time = time.time()
    write_func(file_name, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time to write {file_format} file: {elapsed_time:.2f} seconds")

measure_write_time(df, 'CSV', 'data.csv', df.to_csv, index=False)

measure_write_time(df, 'Parquet', 'data.parquet', df.to_parquet, index=False)

measure_write_time(df, 'HDF5', 'data.h5', df.to_hdf, key='df', mode='w', index=False)
