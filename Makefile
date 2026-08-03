.PHONY: test submit fig-battery \
 test-a submit-a demo-a \
 test-b submit-b demo-b fig-b \
 test-c submit-c demo-c fig-c \
 test-d submit-d demo-d fig-d \
 test-e submit-e demo-e fig-e \
 test-f submit-f demo-f fig-f \
 test-g submit-g demo-g fig-g \
 test-h submit-h demo-h fig-h \
 test-i submit-i demo-i fig-i \
 install clean clean-results clean-datasets \
 use-reference use-student \

# ---- Task A  RNN Module --------------------------------------------------

test-a:
	uv run pytest tests/test_rnn.py tests/test_data.py tests/test_harness.py -q

submit-a:
	uv run python submit.py a

demo-a:
	uv run python scripts/rnn_demo.py --part all

# ---- Task B  Fix the Training Loop ---------------------------------------

test-b:
	uv run pytest tests/test_train.py -q

submit-b:
	uv run python submit.py b

fig-b:
	uv run python scripts/plot.py b

demo-b:
	uv run python scripts/train_rnn.py --part all
	$(MAKE) fig-b
	$(MAKE) fig-battery

# ---- Task C  Additive (Bahdanau) Attention -------------------------------

test-c:
	uv run pytest tests/test_additive.py tests/test_rnn.py -q

submit-c:
	uv run python submit.py c

fig-c:
	uv run python scripts/plot.py c

demo-c:
	uv run python scripts/train_rnn.py --attention --part all
	uv run python scripts/recall_vs_distance.py --part all
	$(MAKE) fig-c
	$(MAKE) fig-battery

# ---- Task D  Scaled Dot-Product Self-Attention ---------------------------

test-d:
	uv run pytest tests/test_scaled_dot_product.py -q

submit-d:
	uv run python submit.py d

fig-d:
	uv run python scripts/plot.py d

demo-d:
	uv run python scripts/probe_battery.py --mechanism self-attn --part all
	uv run python scripts/scaling_ablation.py --part all
	uv run python scripts/permutation_equivariance.py --part all
	$(MAKE) fig-d
	$(MAKE) fig-battery

# ---- Task E  Multi-Head Self-Attention -----------------------------------

test-e:
	uv run pytest tests/test_multihead.py -q

submit-e:
	uv run python submit.py e

fig-e:
	uv run python scripts/plot.py e

demo-e:
	uv run python scripts/probe_battery.py --mechanism multihead --part all
	uv run python scripts/two_relation.py --part all
	$(MAKE) fig-e
	$(MAKE) fig-battery

# ---- Task F  Positional Encoding & ALiBi ---------------------------------

test-f:
	uv run pytest tests/test_positional.py -q

submit-f:
	uv run python submit.py f

fig-f:
	uv run python scripts/plot.py f

demo-f:
	uv run python scripts/length_generalization.py --part all
	$(MAKE) fig-f

# ---- Task G  Decoder-Only Transformer ------------------------------------

test-g:
	uv run pytest tests/test_transformer_decoder.py -q

submit-g:
	uv run python submit.py g

fig-g:
	uv run python scripts/plot.py g

demo-g:
	uv run python scripts/train_transformer.py --part all
	$(MAKE) fig-g
	$(MAKE) fig-battery

# ---- Task H  Encoder + Cross-Attention -----------------------------------

test-h:
	uv run pytest tests/test_transformer_encdec.py -q

submit-h:
	uv run python submit.py h

fig-h:
	uv run python scripts/plot.py h

demo-h:
	uv run python scripts/sort_gap.py --part all
	$(MAKE) fig-h
	$(MAKE) fig-battery

# ---- Task I  Linear (Kernel) Attention -----------------------------------

test-i:
	uv run pytest tests/test_linear.py -q

submit-i:
	uv run python submit.py i

fig-i:
	uv run python scripts/plot.py i

demo-i:
	uv run python scripts/linear_benchmark.py --part all
	$(MAKE) fig-i

# ---- Cross-task figures --------------------------------------------------

# The battery scoreboard spans every mechanism, so it is not task-specific:
# it is regenerated (into results/figures/battery.png) at the end of every
# battery-contributing demo (b, c, d, e, g, h) from whatever rows are on disk.
fig-battery:
	uv run python scripts/plot.py battery

# ---- Utilities -----------------------------------------------------------

submit:
	uv run python submit.py

test:
	uv run pytest tests/ -q

install:
	git init
	uv sync

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

clean-results:
	find results -mindepth 1 -not -type d -not -name '.gitkeep' -delete

clean-datasets:
	find data -mindepth 1 -not -type d -not -name '.gitkeep' -delete

use-reference:
	@if [ -d src_student_backup ]; then \
	    echo "Already in reference mode. Run 'make use-student' first."; \
	    exit 1; \
	elif [ ! -d src_reference ]; then \
	    echo "src_reference/ not found. Run 'make compile-reference' first."; \
	    exit 1; \
	else \
	    mv src src_student_backup && \
	    cp -r src_reference src && \
	    touch .reference_mode && \
	    echo "Reference mode active. Run 'make use-student' to switch back."; \
	fi

use-student:
	@if [ ! -d src_student_backup ]; then \
	    echo "Already in student mode."; \
	else \
	    rm -rf src && \
	    mv src_student_backup src && \
	    rm -f .reference_mode && \
	    echo "Student mode active."; \
	fi

# ---- Instructor-only (stripped at bundle time) ---------------------------

