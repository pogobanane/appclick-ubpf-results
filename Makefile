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
		-o $(OUT_DIR)/reconfiguration_vnfs.pdf \
		-W $(WIDTH) -H 2.5 \
		--1 $(DATA)/reconfiguration_*_rep?.csv --1-name stub

#		--1 $(DATA)/reconfiguration_ukebpfjit_*_rep?.csv --1-name stub



reconfiguration_stack.pdf:
	python3 reconfiguration_stack.py \
		-o $(OUT_DIR)/reconfiguration_stack.pdf \
		-W $(WIDTH) -H 2.5 \
		--1 $(DATA)/reconfiguration_*_rep?.csv --1-name stub

throughput.pdf:
	python3 throughput.py \
		-W $(DWIDTH) -H 3.5 \
		-o $(OUT_DIR)/throughput.pdf \
		--1-name stub --1 $(DATA)/throughput_*_vpp_*.csv

relative_performance.pdf:
	python3 relative_performance.py \
		-o $(OUT_DIR)/relative_performance.pdf \
		-W $(WIDTH) -H 2 \
		--1-name stub --1 $(DATA)/throughput_*_vpp_*.csv \
		--linux-histogram ./data/out7/latency_linux_64B_vpp_mirror_100kpps_rep2.histogram.csv \
		--uk-histogram ./data/out7/latency_uk_64B_vpp_mirror_100kpps_rep2.histogram.csv \

imagesize.pdf:
	python3 imagesize.py \
		-W $(WIDTH) -H 2.5 \
		-o $(OUT_DIR)/imagesize.pdf \
		--1 flake.nix --1-name stub


latency_cdf.pdf:
	python3 latency_cdf.py -W $(WIDTH) -H 2.1 -l \
  	-o $(OUT_DIR)/latency_cdf.pdf \
  	--3-name "Linux: mirror" "l--" "magenta" --3 $(DATA)/latency_linux_64B_vpp_mirror_100kpps_rep?.histogram.csv \
  	--5-name "UniBPF: mirror" "l-" "brown" --5 $(DATA)/latency_ukebpfjit_64B_vpp_mirror_100kpps_rep?.histogram.csv \
  	--8-name "Unikraft: mirror" "l:" "cyan" --8 $(DATA)/latency_uk_64B_vpp_mirror_100kpps_rep?.histogram.csv \

#  	--4-name "Linux: NAT" "l--" "orange" --4 ./data/stub/acc_histogram_pcvm_vmux-dpdk-e810_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \
#  	--7-name "UniBPF: NAT" "l:" "green" --7 ./data/stub/acc_histogram_pcvm_bridge_normal_vhoston_ioregionfdoff_xdp_10kpps_60B_*s.csv \
#  	--9-name "UniBPF NAT" "l:" "violet" --9 ./data/stub/acc_histogram_pcvm_bridge-e1000_normal_vhostoff_ioregionfdoff_xdp_10kpps_60B_*s.csv \

firewall.pdf:
	python3 firewall.py -W $(WIDTH) -H 2.1 -l \
  	-o $(OUT_DIR)/firewall.pdf \
  	--3-name "Linux" "l--" "magenta" --3 $(DATA)/firewall_linux_*.csv \
  	--5-name "Unikraft" "l-" "brown" --5 $(DATA)/firewall_uk_*.csv \
  	--8-name "UniBPF no JIT" "l-" "red" --8 $(DATA)/firewall_ukebpf_*.csv \
  	--9-name "UniBPF" "l:" "cyan" --9 $(DATA)/firewall_ukebpfjit_*.csv \


safety_time.pdf:
	python3 safety_time.py \
		-o $(OUT_DIR)/safety_time.pdf \
		-W $(WIDTH) -H 2.5 \
		--1 flake.nix --1-name stub


