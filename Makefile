WIDTH := 5.0
WIDTH2 := 5.5
DWIDTH := 11
DWIDTH2 := 13

reconfiguration.pdf:
	python3 reconfiguration.py \
		-W $(WIDTH) -H 2.5 \
		--1 flake.nix --1-name stub
