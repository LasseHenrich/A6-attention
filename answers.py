"""Observation answers for A6-attention.

Fill in the values below after reading each task prompt and running its
demo.  Every variable is typed; ``*_observations`` dicts hold booleans (some
statements are deliberately false — judge each one).  ``*_status`` variables
take a value from a small allowed set stated in the prompt.  This is not a
writeup — only these typed values are graded.

The grader hashes your filled values, so leave anything you have not
answered as ``None``.
"""

from __future__ import annotations

# ===========================================================================
# Task A — RNN module
# ===========================================================================

# Architecture facts about the RNN you built (True / False).  Read them off
# the shape/parameter walk (`make demo-a`); some statements are false.
a_observations: dict[str, bool | None] = {
    "encoder_returns_a_hidden_state_for_every_source_token": True,
    "decoder_sees_source_only_through_final_state": True,
    "an_untrained_rnn_scores_at_chance_on_copy": True,
    "the_cell_uses_different_weights_at_each_time_step": False,
}

# The model's total parameter count for the provided default config.  Read it
# from the parameter walk printed by `make demo-a`.
a_param_count: int | None = 20758


# ===========================================================================
# Task B — fix the training loop
# ===========================================================================

# The line numbers of the 5 bugs in the DISTRIBUTED src/attention/train.py.
# Record them BEFORE you start editing (your edits shift the line numbers).
b_bug_lines: list[int] | None = None

# Symptom -> cause reasoning about training-loop bugs (True / False).  Some
# statements are deliberately false.
b_symptom_observations: dict[str, bool | None] = {
    "training_in_eval_mode_can_never_change_results": None,
    "wiping_gradients_after_backward_but_before_step_stops_learning": None,
    "flat_loss_can_come_from_a_missing_or_misplaced_optimizer_step": None,
    "returning_the_loss_tensor_instead_of_item_keeps_the_graph_alive": None,
    "counting_pad_and_prompt_positions_in_the_loss_dilutes_the_signal": None,
}

# The trained RNN's battery profile, read off `make demo-b` (True / False).
b_rnn_observations: dict[str, bool | None] = {
    "the_rnn_solves_sort_perfectly_at_long_lengths": None,
    "the_trained_rnn_beats_an_untrained_one_on_short_tasks": None,
    "the_baseline_rnn_already_uses_attention": None,
    "exact_sequence_tasks_like_copy_are_among_the_hardest_for_the_rnn": None,
}


# ===========================================================================
# Task C — additive (Bahdanau) attention
# ===========================================================================

# From `make demo-c` (recall-vs-distance overlay).  True / False.
c_recall_observations: dict[str, bool | None] = {
    "attention_makes_recall_perfect_at_every_distance": None,
    "attention_improves_recall_over_the_plain_rnn": None,
    "recall_still_decays_as_the_needle_moves_earlier": None,
}

# From `make demo-c` (decoder gradient-norm curve).  True / False.
c_gradient_observations: dict[str, bool | None] = {
    "attention_removes_the_vanishing_gradient_problem": None,
    "gradient_norm_reaching_earlier_source_positions_shrinks": None,
}

# From `make demo-c` (wall-clock vs sequence length).  True / False.
c_serial_observations: dict[str, bool | None] = {
    "attention_makes_the_rnn_parallel_over_time": None,
    "inference_wall_clock_grows_with_sequence_length": None,
}

# Epistemic status of the residual faults (serial + vanishing gradients).
# One of: "empirical" | "provable".
c_fault_status: str | None = None


# ===========================================================================
# Task D — scaled dot-product attention
# ===========================================================================

# From `make demo-d` (scaling ablation).  True / False.
d_scaling_observations: dict[str, bool | None] = {
    "unscaled_attention_saturates_the_softmax": None,
    "scaling_changes_what_the_model_can_represent": None,
    "scaling_restores_useful_gradients": None,
    "unscaled_scores_grow_with_key_dimension": None,
}

# From `make demo-d` (permutation-equivariance panel).  True / False.
d_equivariance_observations: dict[str, bool | None] = {
    "bare_self_attention_can_learn_reverse_with_enough_training": None,
    "a_causal_mask_leaks_position_information": None,
    "permuting_the_input_permutes_the_output": None,
}

# Epistemic status of each fault.  One of: "empirical" | "provable".
d_scaling_status: str | None = None
d_equivariance_status: str | None = None


# ===========================================================================
# Task E — multi-head attention
# ===========================================================================

# From `make demo-e` (single-vs-multi accuracy; per-head entropy).  T / F.
e_head_observations: dict[str, bool | None] = {
    "heads_specialize_to_different_positions": None,
    "multiple_heads_solve_the_two_relation_task": None,
    "a_single_head_plateaus_on_the_two_relation_task": None,
    "more_heads_always_raise_accuracy_on_every_task": None,
}

# From `make demo-e` (order-blindness persists).  True / False.
e_order_observations: dict[str, bool | None] = {
    "multi_head_attention_is_still_order_blind_without_positions": None,
    "adding_heads_gives_the_model_a_sense_of_position": None,
}

# Epistemic status.  One of: "empirical" | "provable".
e_two_relation_status: str | None = None
e_order_status: str | None = None


# ===========================================================================
# Task F — positional encoding & ALiBi
# ===========================================================================

# From `make demo-f` (accuracy vs eval length per scheme).  True / False.
f_length_observations: dict[str, bool | None] = {
    "learned_absolute_pe_cliffs_beyond_the_trained_length": None,
    "learned_absolute_pe_extrapolates_as_well_as_alibi": None,
    "no_pe_fails_order_tasks": None,
    "alibi_extrapolates_to_longer_sequences": None,
}

# From `make demo-f` (ALiBi-vs-sinusoidal recall).  True / False.
f_alibi_observations: dict[str, bool | None] = {
    "alibi_biases_attention_toward_nearby_positions": None,
    "alibi_has_no_downside": None,
    "alibis_locality_prior_can_hurt_long_range_recall": None,
}

# Epistemic status.  One of: "empirical" | "provable".
f_abs_pe_status: str | None = None
f_alibi_status: str | None = None


# ===========================================================================
# Task G — decoder-only transformer + the causal-mask bug
# ===========================================================================

# The line number of the causal-mask bug in the DISTRIBUTED
# src/attention/masking.py.  Record it BEFORE you edit.
g_bug_line: int | None = None

# From `make demo-g`.  True / False.  Read the accuracy-by-position figure
# (buggy | fixed) and the all-heads future-mass grids (leaky | correct).
g_mask_observations: dict[str, bool | None] = {
    "the_buggy_model_has_near_perfect_teacher_forced_accuracy": None,
    "the_buggy_model_collapses_under_free_running_generation": None,
    "the_gap_between_teacher_forced_and_free_running_signals_the_leak": None,
    "teacher_forced_accuracy_alone_would_reveal_the_leak": None,
    "under_the_buggy_mask_free_running_is_at_chance_at_every_output_position": None,
    "the_leak_is_concentrated_in_a_few_heads_rather_than_spread_across_all": None,
    "all_attention_heads_must_be_aligned_for_high_accuracy": None,
    "a_missing_causal_mask_is_an_inductive_bias_fault": None,
}

# The causal-mask leak is categorically different from earlier faults.
# One of: "correctness_bug" | "empirical" | "provable".
g_leak_status: str | None = None


# ===========================================================================
# Task H — encoder + cross-attention
# ===========================================================================

# From `make demo-h`: the printed Sort table (decoder-only vs encoder-decoder),
# h-source-dependency.png, and h-alignment.png.  True / False.
h_sort_observations: dict[str, bool | None] = {
    "the_encoder_decoder_solves_sort_better_than_decoder_only": None,
    "producing_a_sorted_output_requires_having_seen_the_whole_input": None,
    "the_decoder_only_model_sees_the_whole_source_before_emitting_y1": None,
    "the_sort_tie_shows_sorting_does_not_need_a_global_view_of_the_input": None,
    "the_encoder_runs_bidirectionally": None,
    "the_decoder_only_source_dependency_is_exactly_zero_above_the_diagonal": None,
    "only_the_encoder_decoder_can_align_output_positions_to_sorted_order": None,
}

# About cross-attention — definition, plus h-cross-attention-alignment.png
# (Copy / Reverse / Sort).  True / False.
h_crossattn_observations: dict[str, bool | None] = {
    "cross_attention_is_multi_head_bahdanau_attention": None,
    "cross_attention_uses_decoder_queries_and_encoder_keys_and_values": None,
    "cross_attention_is_a_brand_new_mechanism": None,
    "cross_attention_traces_the_argsort_permutation_on_sort": None,
    "cross_attention_traces_the_diagonal_on_copy_and_anti_diagonal_on_reverse": None,
    "cross_attention_weights_are_task_independent": None,
}

# Epistemic status of the two things the encoder changes.  Each is one of:
# "empirical" | "provable".  (Bidirectional source encoding follows from the
# masks; the sharper alignment is a measurement at this scale.)
h_bidirectionality_status: str | None = None
h_alignment_status: str | None = None


# ===========================================================================
# Task I — linear (kernel) attention
# ===========================================================================

# From `make demo-i` (cost/memory vs length).  True / False.
i_cost_observations: dict[str, bool | None] = {
    "linear_attention_cost_grows_more_slowly_with_length": None,
    "softmax_attention_cost_grows_quadratically": None,
    "linear_attention_is_faster_at_short_lengths_too": None,
}

# From `make demo-i` (recall vs load; SNR).  True / False.
i_recall_observations: dict[str, bool | None] = {
    "linear_attention_recall_degrades_as_stored_pairs_grow": None,
    "softmax_recall_holds_up_better_under_load": None,
    "the_retrieval_snr_falls_as_load_grows": None,
    "linear_attention_matches_softmax_recall_at_every_load": None,
}

# Attention entropy is undefined for linear attention (no explicit weights).
i_entropy_na: bool | None = None

# Epistemic status.  One of: "empirical" | "provable".
i_cost_status: str | None = None
i_recall_status: str | None = None
