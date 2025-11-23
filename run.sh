#!/bin/bash
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

# Run Streamlit using the virtual environment python
./.venv/bin/python -m streamlit run app.py --server.fileWatcherType none
