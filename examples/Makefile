# Re-record all reference traces.
#
# Usage:
#   make              — re-record every example
#   make customer_support — re-record a single example
#   make clean        — remove all generated .sb files
#
# Each recipe runs the companion record.py script in the example subdirectory
# with PYTHONPATH set to the repo root so `import stepback` resolves without
# a formal install.

PYTHON := python3
REPO_ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/../
export PYTHONPATH := $(REPO_ROOT)

EXAMPLES := customer_support rag_pipeline tool_using_agent multi_step_planner \
            parallel_branch batch_worker

.PHONY: all $(EXAMPLES) clean

all: $(EXAMPLES)

customer_support:
	$(PYTHON) $(REPO_ROOT)/examples/customer_support/record.py

rag_pipeline:
	$(PYTHON) $(REPO_ROOT)/examples/rag_pipeline/record.py

tool_using_agent:
	$(PYTHON) $(REPO_ROOT)/examples/tool_using_agent/record.py

multi_step_planner:
	$(PYTHON) $(REPO_ROOT)/examples/multi_step_planner/record.py

parallel_branch:
	$(PYTHON) $(REPO_ROOT)/examples/parallel_branch/record.py

batch_worker:
	$(PYTHON) $(REPO_ROOT)/examples/batch_worker/record.py

clean:
	find $(REPO_ROOT)/examples -name "*.sb" -delete
