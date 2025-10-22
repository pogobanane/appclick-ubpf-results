VERSION := v0.0.5
DATA := ./data/$(VERSION)
OUT_DIR := ./out/$(VERSION)

WIDTH := 5.0
WIDTH2 := 5.5
DWIDTH := 11
DWIDTH2 := 13

PAPER_FIGURES := reconfiguration_vnfs.pdf reconfiguration_stack.pdf throughput.pdf imagesize.pdf relative_performance.pdf latency_cdf.pdf firewall.pdf safety_time.pdf mpk.pdf
# PAPER_FIGURES := mediation.pdf microservices.pdf ycsb.pdf
# PAPER_FIGURES := mediation.pdf iperf.pdf ycsb.pdf

all: $(PAPER_FIGURES)

install:
	test -n "$(OVERLEAF)" # OVERLEAF must be set
	for f in $(PAPER_FIGURES); do test -f $(OUT_DIR)/$$f && cp $(OUT_DIR)/$$f $(OVERLEAF)/$$f || true; done

reconfiguration_vnfs.pdf:
	python3 reconfiguration_vnfs.py \
		-o $(OUT_DIR)/reconfiguration_vnfs.pdf \
		-W $(WIDTH) -H 2.2 \
		--1 $(DATA)/reconfiguration_*_rep?.csv --1-name stub

#		--1 $(DATA)/reconfiguration_ukebpfjit_*_rep?.csv --1-name stub



reconfiguration_stack.pdf:
	python3 reconfiguration_stack.py \
		-o $(OUT_DIR)/reconfiguration_stack.pdf \
		-W $(WIDTH) -H 1.8 \
		--1 $(DATA)/reconfiguration_*_rep?.csv --1-name stub

reconfiguration_stack_slides.pdf:
	python3 reconfiguration_stack.py -s \
		-o $(OUT_DIR)/reconfiguration_stack.pdf \
		-W 4.5 -H 2.5 \
		--1 $(DATA)/reconfiguration_*_rep?.csv --1-name stub

throughput.pdf:
	python3 throughput.py \
		-W $(DWIDTH) -H 3 \
		-o $(OUT_DIR)/throughput.pdf \
		--1-name stub --1 ./data/v0.0.4-pre7/throughput_*_vpp_*.csv

throughput_slides.pdf:
	python3 throughput.py \
		-W $(DWIDTH) -H 3 \
		-o $(OUT_DIR)/throughput_slides.pdf \
		--slides \
		--1-name stub --1 ./data/v0.0.4-pre7/throughput_*_vpp_*.csv

relative_performance.pdf:
	python3 relative_performance.py \
		-o $(OUT_DIR)/relative_performance.pdf \
		-W $(WIDTH) -H 1.9 \
		--1-name stub --1 ./data/v0.0.4-pre7/throughput_*_vpp_*.csv \
		--linux-mirror-histogram $(DATA)/latency_linux_64B_vpp_mirror_100kpps_rep?.histogram.csv \
		--uk-mirror-histogram $(DATA)/latency_uk_64B_vpp_mirror_100kpps_rep?.histogram.csv \
		--linux-nat-histogram $(DATA)/latency_linux_64B_vpp_nat_100kpps_rep?.histogram.csv \
		--uk-nat-histogram $(DATA)/latency_uk_64B_vpp_nat_100kpps_rep?.histogram.csv \

imagesize.pdf:
	python3 imagesize.py \
		-W $(WIDTH) -H 1.4 \
		-o $(OUT_DIR)/imagesize.pdf \
		--1 $(DATA)/misc_imagesize_rep0.log --1-name stub


latency_cdf.pdf:
	python3 latency_cdf.py -W $(WIDTH) -H 2.1 \
  	-o $(OUT_DIR)/latency_cdf.pdf \
  	--1-name "Linux: mirror" "l-" "green" --1 $(DATA)/latency_linux_64B_vpp_mirror_100kpps_rep?.histogram.csv \
  	--2-name "MorphOS: mirror" "l-" "blue" --2 $(DATA)/latency_ukebpfjit_64B_vpp_mirror_100kpps_rep?.histogram.csv \
  	--4-name "Linux: NAT" "l:" "orange" --4 $(DATA)/latency_linux_64B_vpp_nat_100kpps_rep?.histogram.csv \
  	--5-name "MorphOS: NAT" "l-" "red" --5 $(DATA)/latency_ukebpfjit_64B_vpp_nat_100kpps_rep?.histogram.csv \
  	--6-name "Unikraft: mirror" "l:" "cyan" --6 $(DATA)/latency_uk_64B_vpp_mirror_100kpps_rep?.histogram.csv \
  	--7-name "Unikraft: NAT" "l:" "purple" --7 $(DATA)/latency_uk_64B_vpp_nat_100kpps_rep?.histogram.csv \


firewall.pdf:
	python3 firewall.py -W 4 -H 1.7 -l \
  	-o $(OUT_DIR)/firewall.pdf \
  	--3-name "Linux" "l--" "magenta" --3 ./data/v0.0.4-pre7/firewall_linux_*.csv \
  	--5-name "Unikraft" "l-" "brown" --5 ./data/v0.0.4-pre7/firewall_uk_*.csv \
  	--8-name "MorphOS no JIT" "l-" "red" --8 ./data/v0.0.4-pre7/firewall_ukebpf_*.csv \
  	--9-name "MorphOS" "l:" "cyan" --9 ./data/v0.0.4-pre7/firewall_ukebpfjit_*.csv \


mpk.pdf:
	python3 mpk.py -W 4 -H 1.5 \
  	-o $(OUT_DIR)/mpk.pdf \
		--2-name "MorphOS + MPK" --2 ./data/v0.0.4-pre7/throughput_ukebpfjit_vpp_rx_filter_*.csv \
		--1-name "MorphOS" --1 ./data/v0.0.4-pre7/throughput_ukebpfjit_nompk_vpp_rx_filter_*.csv #\
		#--3-name "Max IO bandwidth" --3 flake.nix

mpk-bars.pdf:
	python3 mpk-bars.py -W $(WIDTH) -H 2.1 \
  	-o $(OUT_DIR)/mpk-bars.pdf \
		--1 flake.nix --1-name stub


safety_time.pdf:
	python3 safety_time.py \
		-o $(OUT_DIR)/safety_time.pdf \
		-W $(WIDTH) -H 2.5 \
		--bpfbuild $(DATA)/misc_safetytime_rep0.log \
		--1 $(DATA)/reconfiguration_*_rep?.csv --1-name stub


