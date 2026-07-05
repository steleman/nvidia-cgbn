SUBDIRS := sample_01_add sample_02_modinv sample_03_powm sample_04_miller_rabin sample_05_signed sample_06_large_div

TARGETS := clean kepler maxwell pascal volta turing ampere ada blackwell

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo "   make turing"
	@echo "   make ampere"
	@echo "   make ada"
	@echo "   make blackwell"
	@echo

$(TARGETS): $(SUBDIRS)
$(SUBDIRS):
	make -C $@ $(MAKECMDGOALS)

.PHONY: $(TARGETS) $(SUBDIRS)

