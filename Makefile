SHELL:=/usr/bin/bash

default: build test

##############################################################################
#
# Build
#
##############################################################################

# Recreate Makefiles when CMakeLists.txt changes
./build/debug/Makefile: CMakeLists.txt
	mkdir -p ./build/debug
	mkdir -p ./build/release
	cd build/debug && cmake -D CMAKE_BUILD_TYPE=Debug ../..
	cd build/release && cmake -D CMAKE_BUILD_TYPE=Release ../..

.PHONY: build # Build all targets
build: ./build/debug/Makefile
	@cd build/debug && make -j
	@cd build/release && make -j

.PHONY: clean # Remove all build dependencies
clean:
	@rm -rf build

##############################################################################
#
# Test
#
##############################################################################

.PHONY: unit_test
unit_test: BUILD=debug
unit_test:
	@parallel --jobs 24 --halt now,fail=1 "echo {} && {}" ::: build/$(BUILD)/test_*

.PHONY: test # Run tests
test:
	@echo "Testing..."
	@$(MAKE) --no-print-directory unit_test BUILD=debug
	@$(MAKE) --no-print-directory unit_test BUILD=release

##############################################################################
#
# Machine Learning
#
##############################################################################

INPUT=./data/remote/latest/*.csv
MODEL=./models/model.json
BUILD=release
EPOCHS=100

.PHONY: train # Train a model
train: build
	@find $(INPUT) | ./build/$(BUILD)/train \
		--verbose \
		--balance-priors-ratio=3 \
		--random-seed=123 \
		--epochs=$(EPOCHS) \
		--output-model-filename=$(MODEL)

.PHONY: search # Search for the best feature parameters
search: build
	@find $(INPUT) | ./build/$(BUILD)/train \
		--verbose \
		--balance-priors-ratio=3 \
		--random-seed=123 \
		--epochs=$(EPOCHS) \
		--search

.PHONY: create_features # Create features for tuning hyper-parameters
create_features: build
	@find $(INPUT) | ./build/$(BUILD)/train \
		--verbose \
		--balance-priors-ratio=3 \
		--random-seed=123 \
		--epochs=$(EPOCHS) \
		--feature-dump-filename features.csv

.PHONY: tune_hyperparams # Tune hyper-parameters
tune_hyperparams:
	@./scripts/tune_hyperparams.py \
		--verbose \
		features.csv

.PHONY: classify # Run classifier
classify: build
	@mkdir -p ./predictions
	@parallel --verbose --lb --jobs=15 --halt now,fail=1 \
		"./build/$(BUILD)/classify \
			--verbose \
			--model-filename=$(MODEL) \
			< {} \
			> predictions/{/.}_classified.csv" \
	::: $(INPUT)

.PHONY: score # Score results
score:
	@parallel --verbose --lb --jobs=15 --halt now,fail=1 \
		"./build/$(BUILD)/score \
			--verbose \
			< predictions/{/.}_classified.csv \
			> predictions/{/.}_score.txt" \
	::: $(INPUT)
	@./scripts/summarize_scores.bash | tee scores.txt

.PHONY: cross_val # Cross validate
cross_val:
	@./scripts/generate_cross_val_commands.py \
		--verbose \
		--splits=5 \
		"$(INPUT)" > ./predictions/cross_validate.bash
	@cat ./predictions/cross_validate.bash
	@bash ./predictions/cross_validate.bash
	@rm ./predictions/cross_validate.bash

##############################################################################
#
# View results
#
##############################################################################

.PHONY: view # View predictions
view:
	@parallel --lb --jobs=100 \
		"streamlit run ../ATL24_viewer/view_predictions.py -- --verbose {}" \
		::: $$(find ./predictions/*.csv | head)

##############################################################################
#
# Get help by running
#
#     $ make help
#
##############################################################################
.PHONY: help # Generate list of targets with descriptions
help:
	@grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1	\2/' | expand -t25
