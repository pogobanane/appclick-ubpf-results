OUT_DIR := ./
DATA := ./data/v0.0.3-pre

WIDTH := 5.0
WIDTH2 := 5.5
DWIDTH := 11
DWIDTH2 := 13

PAPER_FIGURES := reconfiguration_vnfs.pdf reconfiguration_stack.pdf throughput.pdf imagesize.pdf relative_performance.pdl
# PAPER_FIGURES := mediation.pdf microservices.pdf ycsb.pdf
# PAPER_FIGURES := mediation.pdf iperf.pdf ycsb.pdf

all: $(PAPER_FIGURES)

install:
	test -n "$(OVERLEAF)" # OVERLEAF must be set
	for f in $(PAPER_FIGURES); do test -f $(OUT_DIR)/$$f && cp $(OUT_DIR)/$$f $(OVERLEAF)/$$f || true; done

reconfiguration_vnfs.pdf:
	python3 reconfiguration_vnfs.py \
		-W $(WIDTH) -H 2.5 \
		--1 flake.nix --1-name stub

reconfiguration_stack.pdf:
	python3 reconfiguration_stack.py \
		-W $(WIDTH) -H 2.5 \
		--1 flake.nix --1-name stub

throughput.pdf:
	python3 throughput.py \
		-W $(WIDTH2) -H 2.5 \
		--1-name stub --1 $(DATA)/throughput_*_vpp_*.csv

relative_performance.pdf:
	python3 relative_performance.py \
		-W $(WIDTH2) -H 2 \
		--1-name stub --1 $(DATA)/throughput_*_vpp_*.csv

imagesize.pdf:
	python3 imagesize.py \
		-W $(WIDTH) -H 2.5 \
		--1 flake.nix --1-name stub
