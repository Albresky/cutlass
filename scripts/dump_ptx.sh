#!/bin/bash

# --- Configuration ---
TARGET_ARCH_NUM=100
CUTLASS_ROOT="/root/wkspace/cutlass"
# --- End of Configuration ---

export PATH=/usr/local/cuda-13.0/bin:$PATH

# Derived paths and flags
EXAMPLES_DIR="${CUTLASS_ROOT}/examples"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE_DIR="./dump_ptx/results_${TIMESTAMP}"
ARCH_FLAG="arch=compute_${TARGET_ARCH_NUM},code=sm_${TARGET_ARCH_NUM}"

# Create the main results directory and the report file
mkdir -p ${OUTPUT_BASE_DIR}
FINAL_REPORT_FILE="$(readlink -f ${OUTPUT_BASE_DIR}/mapping_summary.txt)"

touch "${FINAL_REPORT_FILE}"
echo "Created base output directory: ${OUTPUT_BASE_DIR}"
echo "Final report will be generated at: ${FINAL_REPORT_FILE}"
echo "Compiling for SM=${TARGET_ARCH_NUM}"
echo "====================================================="

# Find and process numbered example directories
find "${EXAMPLES_DIR}" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' | sort -V | while read -r example_dir; do
    
    EXAMPLE_DIR_NAME=$(basename "${example_dir}")
    echo "Processing Example Directory: ${EXAMPLE_DIR_NAME}"

    find "${example_dir}" -name "*.cu" | while read -r cu_file; do
        
        CU_BASE_NAME=$(basename "${cu_file}" .cu)
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXAMPLE_DIR_NAME}/${CU_BASE_NAME}"
        echo "output_dir: ${OUTPUT_DIR}"

        # get the dir of cu_file
        CUR_DIR=$(dirname "${cu_file}")
        echo "current dir: $CUR_DIR"
        
        echo "  -> Compiling: ${CU_BASE_NAME}.cu"
        mkdir -p "${OUTPUT_DIR}"

        pushd "${OUTPUT_DIR}" > /dev/null

        # Compile the .cu file
        if ! nvcc -std=c++17 \
            --expt-relaxed-constexpr \
            -I "${CUR_DIR}" \
            -I "${CUTLASS_ROOT}/include" \
            -I "${CUTLASS_ROOT}/include/cute/include" \
            -I "${CUTLASS_ROOT}/tools/util/include"  \
            -I "${CUTLASS_ROOT}/tools/profiler/include"  \
            -I "${CUTLASS_ROOT}/tools/library/include" \
            -I "${CUTLASS_ROOT}/examples/common" \
            -c "${cu_file}" \
            -o "${CU_BASE_NAME}.cu.o" \
            -gencode "${ARCH_FLAG}" \
            --keep 2>&1 | tee "compile.log"; then
            echo "    -> ❌ FAILED to compile. See log."
            popd > /dev/null
            continue
        fi
        echo "    -> ✅ Compiled successfully."
        echo "    -> Analyzing kernels and generating PTX signatures..."


        # === CORE ANALYSIS LOGIC ===
        # 1. Get a map of mangled names to readable names
        nm *.cu.o | grep "__device_stub__" | grep " t " |awk '{print $3}' > mangled_names.txt
        nm *.cu.o | grep "__device_stub__" | c++filt | grep " t " > readable_names.txt
        grep "// .globl" *.ptx | awk '{print $3}' | c++filt > knl_symbols_from_ptx.log

        paste mangled_names.txt readable_names.txt > kernel_map.txt

        # 2. Process each kernel found
        while read -r mangled_name readable_name_line; do
            
            # Extract the core mangled name from the stub
            core_mangled_name=$(echo "${mangled_name}" | sed 's/.*__device_stub__//')

            # Extract the readable C++ signature
            readable_signature=$(echo "${readable_name_line}" | sed 's/.*__device_stub__//; s/^ *[a-zA-Z0-9_]* *//')

            # Write kernel header to the final report
            echo "====================================================================================" >> "${FINAL_REPORT_FILE}"
            echo "Kernel: ${readable_signature}" >> "${FINAL_REPORT_FILE}"
            echo "File: ${EXAMPLE_DIR_NAME}/${CU_BASE_NAME}.cu" >> "${FINAL_REPORT_FILE}"
            echo "------------------------------------------------------------------------------------" >> "${FINAL_REPORT_FILE}"

            # Extract the PTX block for this specific kernel
            # Then, extract just the instructions and modifiers, sort uniquely, and append to the report
            awk "/\.entry ${core_mangled_name}/,/\\}/" *.ptx | \
            grep -oE '^\s+[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*' | \
            sed 's/^\s*//' | \
            sort -u >> "${FINAL_REPORT_FILE}"
            
            echo "" >> "${FINAL_REPORT_FILE}"

        done < kernel_map.txt
        # === END OF ANALYSIS LOGIC ===
        
        echo "    -> ✅ Analysis complete."
        popd > /dev/null
    done
    echo "-----------------------------------------------------"
done

echo "All tasks finished."
