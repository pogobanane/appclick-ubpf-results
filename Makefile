OUT_DIR := ./

WIDTH := 5.0
WIDTH2 := 5.5
DWIDTH := 11
DWIDTH2 := 13

PAPER_FIGURES := reconfiguration.pdf throughput.pdf imagesize.pdf
# PAPER_FIGURES := mediation.pdf microservices.pdf ycsb.pdf
# PAPER_FIGURES := mediation.pdf iperf.pdf ycsb.pdf

all: $(PAPER_FIGURES)

install:
	test -n "$(OVERLEAF)" # OVERLEAF must be set
	for f in $(PAPER_FIGURES); do test -f $(OUT_DIR)/$$f && cp $(OUT_DIR)/$$f $(OVERLEAF)/$$f || true; done

reconfiguration.pdf:
	python3 reconfiguration.py \
		-W $(WIDTH) -H 2.5 \
		--1 flake.nix --1-name stub

throughput.pdf:
	python3 throughput.py \
		-W $(WIDTH2) -H 2.5 \
		--1 flake.nix --1-name stub

imagesize.pdf:
	python3 imagesize.py \
		-W $(WIDTH) -H 2.5 \
		--1 flake.nix --1-name stub
