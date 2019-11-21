NVCC = nvcc
FLAGS_CPP = -std=c++11 -pthread
FLAGS_CU = -std=c++11
FLAGS_AVX = -mavx -mavx2
MKDIR = mkdir -p

USE_CUDA := $(shell command -v nvcc 2> /dev/null)
USE_AVX := $(shell grep avx2 /proc/cpuinfo)

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	OUTPUT_DIR = bin/debug
	FLAGS_CPP += -g -Og
	FLAGS_CU += -g -O0
else
	OUTPUT_DIR = bin/release
	FLAGS_CPP += -O3
	FLAGS_CU += -O3
endif

# Executable files
EXECUTABLES += sketch
EXECUTABLES += sketch_multithread

ifdef USE_AVX
	# EXECUTABLES_AVX += sketch_avx
	# EXECUTABLES_AVX += sketch_avx_multithread
	# EXECUTABLES_AVX += sketch_avx_multithread_approx
	EXECUTABLES_AVX += sketch_avx_pipelined
endif

ifdef USE_CUDA
	EXECUTABLES_CU += sketch_cu
	# EXECUTABLES_CU += sketch_cu_pipelined
	# EXECUTABLES_CU += sketch_cu_approx
endif

EXECUTABLES := $(addprefix $(OUTPUT_DIR)/, $(EXECUTABLES))
EXECUTABLES_AVX := $(addprefix $(OUTPUT_DIR)/, $(EXECUTABLES_AVX))
EXECUTABLES_CU := $(addprefix $(OUTPUT_DIR)/, $(EXECUTABLES_CU))

OFILES := fasta.o MappedFile.o
OFILES := $(addprefix $(OUTPUT_DIR)/, $(OFILES))

# Rules
all: $(EXECUTABLES) $(EXECUTABLES_AVX) $(EXECUTABLES_CU)

clean:
	rm -r bin/

$(OFILES): $(OUTPUT_DIR)/%.o: src/%.cpp
	@$(MKDIR) $(OUTPUT_DIR)
	$(CXX) $(FLAGS_CPP) -c $^ -o $@

$(EXECUTABLES): $(OUTPUT_DIR)/%: src/%.cpp $(OFILES)
	@$(MKDIR) $(OUTPUT_DIR)
	$(CXX) $(FLAGS_CPP) $^ -o $@

$(EXECUTABLES_AVX): $(OUTPUT_DIR)/%: src/%.cpp $(OFILES)
	@$(MKDIR) $(OUTPUT_DIR)
	$(CXX) $(FLAGS_CPP) $(FLAGS_AVX) $^ -o $@

$(EXECUTABLES_CU): $(OUTPUT_DIR)/%_cu: src/%.cu $(OFILES)
	@$(MKDIR) $(OUTPUT_DIR)
	$(NVCC) $(FLAGS_CU) --compiler-options="$(FLAGS_AVX) $(FLAGS_CPP)" $^ -o $@
