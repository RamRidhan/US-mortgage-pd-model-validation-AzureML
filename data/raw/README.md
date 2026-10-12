# data/raw (not in the repository)

Raw files are never committed (`.gitignore`). To reproduce, download them yourself:

1. Register (free) and download the **Single-Family Loan-Level Dataset, Standard Dataset sample** from Freddie Mac (Clarity data site, search "Freddie Mac Single-Family Loan-Level Dataset"). Accept their terms: loan-level files may not be redistributed.
2. Place the origination and performance files here, named `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt` for 2012 to 2018.
3. Download FRED series `UNRATE` (unemployment rate) and `USSTHPI` as CSV from fred.stlouisfed.org and save them as `UNRATE.csv` and `USSTHPI.csv`.
4. Keep the Freddie Mac User Guide PDF nearby: the file layout (31 origination fields, 35 performance fields in these files) was checked against it, and a column-count assertion fails loudly if a release changes it.
