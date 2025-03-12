VERSION := v0.0.3-pre4
DATA := ./data/$(VERSION)
OUT_DIR := ./out/$(VERSION)

WIDTH := 5.0
WIDTH2 := 5.5
DWIDTH := 11
DWIDTH2 := 13

PAPER_FIGURES := reconfiguration_vnfs.pdf reconfiguration_stack.pdf throughput.pdf imagesize.pdf relative_performance.pdf latency_cdf.pdf firewall.pdf safety_time.pdf
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
		-W $(DWIDTH) -H 3.5 \
		-o $(OUT_DIR)/throughput.pdf \
		--1-name stub --1 $(DATA)/throughput_*_vpp_*.csv

relative_performance.pdf:
	python3 relative_performance.py \
		-W $(WIDTH2) -H 2 \
		--1-name stub --1 $(DATA)/throughput_*_vpp_*.csv

imagesize.pdf:
	python3 imagesize.py \
		-W $(WIDTH) -H 2.5 \
		-o $(OUT_DIR)/imagesize.pdf \
		--1 flake.nix --1-name stub


latency_cdf.pdf:
	python3 latency_cdf.py -W $(WIDTH) -H 2.1 -l \
  	-o $(OUT_DIR)/latency_cdf.pdf \
  	--3-name "Linux: empty" "l--" "magenta" --3 ./data/stub/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--4-name "Linux: NAT" "l--" "orange" --4 ./data/stub/acc_histogram_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--5-name "UniBPF: empty" "l-" "brown" --5 ./data/stub/acc_histogram_pcvm_vmux-med_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--7-name "UniBPF: NAT" "l:" "green" --7 ./data/stub/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--8-name "UniBPF no JIT: empty" "l:" "cyan" --8 ./data/stub/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--9-name "UniBPF no JIT: NAT" "l:" "violet" --9 ./data/stub/acc_histogram_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \

firewall.pdf:
	python3 firewall.py -W $(WIDTH) -H 2.1 -l \
  	-o $(OUT_DIR)/firewall.pdf \
  	--3-name "Linux" "l--" "magenta" --3 ./data/stub/acc_histogram_pcvm_vmux-emu_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--5-name "UniBPF" "l-" "brown" --5 ./data/stub/acc_histogram_pcvm_vmux-med_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
  	--8-name "UniBPF no JIT" "l:" "cyan" --8 ./data/stub/acc_histogram_pcvm_bridge_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \


safety_time.pdf:
	python3 safety_time.py \
		-o $(OUT_DIR)/safety_time.pdf \
		-W $(WIDTH) -H 2.5 \
		--1 flake.nix --1-name stub


