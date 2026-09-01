#!/usr/bin/env bash
# ============================================================
# Loop Qwen3-ASR WER evaluation over multiple experiments and
# checkpoint ranges.
#
# For each experiment you provide:
#   - gt-csv               : ground-truth CSV. Can be passed multiple times
#                            to share the rest of the config across several
#                            CSVs, e.g.
#                              --gt-csv /path/to/a.csv \
#                              --gt-csv /path/to/b.csv \
#                              --eval-dir ... --tgt-dir ... ...
#                            Each CSV is then looped over the full ckpt range.
#
#                            Optional row-slice suffix (after the path):
#                              path/to/x.csv@@_first_n_=100
#                                  -> only the first 100 CSV rows are evaluated;
#                                     results go to  <tgt-dir>/<csv_name>_first100/iter_xxx
#                              path/to/x.csv@@_last_n_=100
#                                  -> only the last 100 CSV rows are evaluated;
#                                     results go to  <tgt-dir>/<csv_name>_last100/iter_xxx
#                            Without a suffix, all rows are evaluated and
#                            output goes to <tgt-dir>/<csv_name>/iter_xxx as before.
#                            The INPUT dir always uses the ORIGINAL csv_name
#                            (videos/audios were generated using the full CSV).
#   - eval-dir             : root sample dir (parent of iter_xxxxxxx)
#   - tgt-dir              : root output dir (parent of <csv_name>)
#   - ckpt-start           : starting iteration (inclusive)
#   - ckpt-end             : ending iteration (inclusive)
#   - ckpt-interval        : step size (always positive; direction is auto)
#   - caption-column       : CSV column with the caption text
#   - index-column         : CSV column with the sample index
#   - suffix               : suffix appended to index to form filename stem
#   - is-extract-ref-text  : 'true' to extract <speech>/<lyrics> tag content
#                            from the caption as reference; 'false' to use
#                            the whole caption text as the reference.
#   - force-rerun          : 'true' to re-run a ckpt even if its
#                            wer_results.json already exists; 'false' to
#                            skip ckpts whose results already exist
#                            (default 'false'). Can also be forced globally
#                            via env: FORCE_RERUN=1 bash eval_wer.sh
#
# Per-checkpoint, the actual input directory is auto-detected as:
#   <eval-dir>/iter_<ckpt>/<csv_name>/videos     (preferred)
#   <eval-dir>/iter_<ckpt>/<csv_name>/audios     (fallback)
# If neither exists, the ckpt is skipped and NO output dir is created.
#
# The per-checkpoint output directory is:
#   <tgt-dir>/<csv_name>/iter_<ckpt>
# where <csv_name> = basename(gt-csv) without the .csv extension.
# A ckpt is considered already-done when
#   <tgt-dir>/<csv_name>/iter_<ckpt>/wer_results.json
# exists; such ckpts are skipped unless force-rerun is enabled.
#
# Default runtime: single-node, 8 GPUs via torchrun.
# ============================================================

# WER推理的transformers版本
#pip install transformers==4.57.6
#pip install jiwer
#pip install nagisa

# 训练的transformers版本
#pip install transformers==5.5.4

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_DEFAULT_QWEN_PATH="/apdcephfs_wzd2/share_305640887/1_public_models/hymm_ar_assets/WER/Qwen3-ASR-1.7B"
BATCH_SIZE=32
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# Global override: FORCE_RERUN=1 forces re-running every ckpt regardless of
# whether its results already exist (overrides per-experiment --force-rerun).
FORCE_RERUN=${FORCE_RERUN:-0}

_truthy() {
    case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|t|yes|y) return 0 ;;
        *) return 1 ;;
    esac
}

# Parse one --gt-csv entry of the form:
#   path/to/x.csv
#   path/to/x.csv@@_first_n_=N
#   path/to/x.csv@@_last_n_=N
# Output (via globals; bash lacks tuple returns):
#   _GT_CSV_PATH        : raw csv path (left of @@)
#   _GT_CSV_SLICE_MODE  : "" | "first" | "last"
#   _GT_CSV_SLICE_N     : "" | <int>
# Returns non-zero on malformed slice spec.
_parse_gt_csv_entry() {
    local entry="$1"
    local path slice_spec
    if [[ "${entry}" == *"@@"* ]]; then
        path="${entry%%@@*}"
        slice_spec="${entry#*@@}"
    else
        path="${entry}"
        slice_spec=""
    fi

    _GT_CSV_PATH="${path}"
    _GT_CSV_SLICE_MODE=""
    _GT_CSV_SLICE_N=""

    if [[ -z "${slice_spec}" ]]; then
        return 0
    fi

    if [[ "${slice_spec}" =~ ^_first_n_=([0-9]+)$ ]]; then
        _GT_CSV_SLICE_MODE="first"
        _GT_CSV_SLICE_N="${BASH_REMATCH[1]}"
    elif [[ "${slice_spec}" =~ ^_last_n_=([0-9]+)$ ]]; then
        _GT_CSV_SLICE_MODE="last"
        _GT_CSV_SLICE_N="${BASH_REMATCH[1]}"
    else
        echo "[ERROR] invalid gt-csv slice spec '@@${slice_spec}' in '${entry}'" >&2
        echo "        expected '@@_first_n_=N' or '@@_last_n_=N'" >&2
        return 2
    fi
    return 0
}

# ============================================================
# Per-experiment runner.
# Usage: see CLI flags below.
# ============================================================
run_experiment() {
    local -a gt_csvs=()
    local eval_dir=""
    local tgt_dir=""
    local ckpt_start=""
    local ckpt_end=""
    local ckpt_interval=""
    local caption_column="prompt"
    local index_column="index"
    local suffix=""
    local is_extract_ref_text="true"
    local force_rerun="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --gt-csv)               gt_csvs+=("$2");           shift 2 ;;
            --eval-dir)             eval_dir="$2";             shift 2 ;;
            --tgt-dir)              tgt_dir="$2";              shift 2 ;;
            --ckpt-start)           ckpt_start="$2";           shift 2 ;;
            --ckpt-end)             ckpt_end="$2";             shift 2 ;;
            --ckpt-interval)        ckpt_interval="$2";        shift 2 ;;
            --caption-column)       caption_column="$2";       shift 2 ;;
            --index-column)         index_column="$2";         shift 2 ;;
            --suffix)               suffix="$2";               shift 2 ;;
            --is-extract-ref-text)  is_extract_ref_text="$2";  shift 2 ;;
            --force-rerun)          force_rerun="$2";          shift 2 ;;
            *) echo "[ERROR] run_experiment: unknown arg '$1'" >&2; return 2 ;;
        esac
    done

    # Global env override wins over per-experiment setting.
    local effective_force="${force_rerun}"
    if _truthy "${FORCE_RERUN}"; then
        effective_force="true"
    fi

    if (( ${#gt_csvs[@]} == 0 )); then
        echo "[ERROR] run_experiment: missing required arg --gt-csv (can be repeated)" >&2
        return 2
    fi
    for v in eval_dir tgt_dir ckpt_start ckpt_end ckpt_interval; do
        if [[ -z "${!v}" ]]; then
            echo "[ERROR] run_experiment: missing required arg --${v//_/-}" >&2
            return 2
        fi
    done

    if (( ckpt_interval <= 0 )); then
        echo "[ERROR] --ckpt-interval must be a positive integer, got ${ckpt_interval}" >&2
        return 2
    fi

    local step
    if (( ckpt_start >= ckpt_end )); then
        step=$(( -ckpt_interval ))
    else
        step=$(( ckpt_interval ))
    fi

    echo
    echo "############################################################"
    echo "# Experiment group (${#gt_csvs[@]} CSV(s) sharing config)"
    echo "#   eval_dir             = ${eval_dir}"
    echo "#   tgt_dir              = ${tgt_dir}"
    echo "#   ckpt range           = ${ckpt_start} -> ${ckpt_end}  step=${step}"
    echo "#   caption_column       = ${caption_column}"
    echo "#   index_column         = ${index_column}"
    echo "#   suffix               = '${suffix}'"
    echo "#   is_extract_ref_text  = ${is_extract_ref_text}"
    echo "#   force_rerun          = ${effective_force} (per-exp=${force_rerun}, env FORCE_RERUN=${FORCE_RERUN})"
    echo "#   gt_csvs              ="
    local _gc
    for _gc in "${gt_csvs[@]}"; do
        echo "#     - ${_gc}"
    done
    echo "############################################################"

    local gt_entry gt_csv csv_basename original_csv_name effective_csv_name
    local slice_mode slice_n
    for gt_entry in "${gt_csvs[@]}"; do
        if ! _parse_gt_csv_entry "${gt_entry}"; then
            echo "[WARN] skipping malformed --gt-csv '${gt_entry}'" >&2
            continue
        fi
        gt_csv="${_GT_CSV_PATH}"
        slice_mode="${_GT_CSV_SLICE_MODE}"
        slice_n="${_GT_CSV_SLICE_N}"

        csv_basename=$(basename "${gt_csv}")
        original_csv_name="${csv_basename%.*}"
        if [[ -n "${slice_mode}" ]]; then
            effective_csv_name="${original_csv_name}_${slice_mode}${slice_n}"
        else
            effective_csv_name="${original_csv_name}"
        fi

        # Build the python --first-n / --last-n flags from the slice spec.
        local -a slice_args=()
        case "${slice_mode}" in
            first) slice_args=(--first-n "${slice_n}") ;;
            last)  slice_args=(--last-n  "${slice_n}") ;;
            "")    : ;;
        esac

        echo
        echo "============================================================"
        echo "[CSV] ${effective_csv_name}"
        echo "      gt_csv             = ${gt_csv}"
        if [[ -n "${slice_mode}" ]]; then
            echo "      row slice          = ${slice_mode} ${slice_n}"
            echo "      input csv_name     = ${original_csv_name}    (videos/audios use full CSV name)"
            echo "      output subdir name = ${effective_csv_name}"
        fi
        echo "============================================================"

        local ckpt=$ckpt_start
        while :; do
            local iter_str
            iter_str=$(printf "iter_%07d" "${ckpt}")

            # Input dir is always keyed by the ORIGINAL csv name because that
            # is the directory the sample-generation step wrote to.
            local base_dir="${eval_dir}/${iter_str}/${original_csv_name}"
            local input_dir=""
            if [[ -d "${base_dir}/videos" ]]; then
                input_dir="${base_dir}/videos"
            elif [[ -d "${base_dir}/audios" ]]; then
                input_dir="${base_dir}/audios"
            fi

            # Output dir uses the EFFECTIVE csv name so that different slices
            # of the same CSV land in distinct folders.
            local out_dir="${tgt_dir}/${effective_csv_name}/${iter_str}"
            local result_file="${out_dir}/wer_results.json"

            if [[ -z "${input_dir}" ]]; then
                # Input not yet generated for this ckpt: skip silently without
                # creating any output dir/files.
                echo
                echo "[Skip:missing-input] ${effective_csv_name}/${iter_str}: neither "
                echo "    '${base_dir}/videos' nor '${base_dir}/audios' exists; no output written."
            elif [[ -f "${result_file}" ]] && ! _truthy "${effective_force}"; then
                echo
                echo "[Skip:already-done] ${effective_csv_name}/${iter_str}: ${result_file} already exists."
                echo "    Set --force-rerun true (per experiment) or FORCE_RERUN=1 (env) to re-run."
            else
                mkdir -p "${out_dir}"

                echo
                echo "------------------------------------------------------------"
                if [[ -f "${result_file}" ]]; then
                    echo "[Run:force]  csv=${effective_csv_name}  ckpt=${ckpt}  (overwriting existing results)"
                else
                    echo "[Run]        csv=${effective_csv_name}  ckpt=${ckpt}"
                fi
                if [[ -n "${slice_mode}" ]]; then
                    echo "      slice  = ${slice_mode} ${slice_n}"
                fi
                echo "      input  = ${input_dir}"
                echo "      output = ${out_dir}"
                echo "------------------------------------------------------------"

                torchrun --nproc_per_node="${NPROC_PER_NODE}" "${SCRIPT_DIR}/eval_wer.py" \
                    --gt-csv "${gt_csv}" \
                    --eval-dir "${input_dir}" \
                    --tgt-dir "${out_dir}" \
                    --caption-column "${caption_column}" \
                    --index-column "${index_column}" \
                    --suffix "${suffix}" \
                    --is-extract-ref-text "${is_extract_ref_text}" \
                    --model-path "${_DEFAULT_QWEN_PATH}" \
                    --batch-size "${BATCH_SIZE}" \
                    ${slice_args[@]+"${slice_args[@]}"}
                local rc=$?
                if (( rc != 0 )); then
                    echo "[WARN] torchrun exited with code ${rc} for ${effective_csv_name}/${iter_str}; continuing." >&2
                fi
            fi

            # advance ckpt (inclusive of ckpt_end)
            if (( ckpt == ckpt_end )); then
                break
            fi
            local next=$(( ckpt + step ))
            if (( step < 0 )); then
                if (( next < ckpt_end )); then break; fi
            else
                if (( next > ckpt_end )); then break; fi
            fi
            ckpt=${next}
        done
    done
}

# ============================================================
# Experiment configurations
#   Add / remove / edit `run_experiment` calls below. Each call
#   runs sequentially over its full ckpt range.
# ============================================================

run_experiment \
    --gt-csv              "../../../data/test_video/t2va_single_captions_300_5s_15s.csv@@_first_n_=100" \
    --gt-csv              "../../../data/test_video/t2va_multi_captions_300_5s_15s.csv@@_first_n_=100" \
    --gt-csv              "../../../data/test_video/i2va_single_captions_300_5s_15s.csv@@_first_n_=100" \
    --gt-csv              "../../../data/test_video/i2va_multi_captions_300_5s_15s.csv@@_first_n_=100" \
    --gt-csv              "../../../data/test_video/fl2va_single_captions_300_5s_15s.csv@@_first_n_=100" \
    --gt-csv              "../../../data/test_video/fl2va_multi_captions_300_5s_15s.csv@@_first_n_=100" \
    --eval-dir            "/apdcephfs_wzd2/share_305640887/hunyuan/jarvizhang/basic_train_jarvizhang_20260522022116_46e15478/95a68c749e885016019e89287b150084/samples_metrics" \
    --tgt-dir             "/apdcephfs_wzd2/share_305640887/hunyuan/jarvizhang/basic_train_jarvizhang_20260522022116_46e15478/95a68c749e885016019e89287b150084/samples_metrics/asr_results_test" \
    --ckpt-start          20000 \
    --ckpt-end            24000 \
    --ckpt-interval       1000 \
    --caption-column      prompt \
    --index-column        index \
    --suffix              _0 \
    --is-extract-ref-text true \
    --force-rerun         false

# ------------------------------------------------------------
# Example: share the SAME config across multiple CSVs by repeating
# --gt-csv. Each CSV loops over the full ckpt range; the rest of the
# config (eval-dir, tgt-dir, ckpt range, columns, suffix, flags) is
# shared. No need to copy/paste the whole block per CSV.
#
# A CSV path may optionally carry a row-slice suffix:
#   path/to/x.csv@@_first_n_=100   -> only first 100 rows; output goes to
#                                     <tgt-dir>/<csv_name>_first100/iter_xxx
#   path/to/x.csv@@_last_n_=100    -> only last 100 rows;  output goes to
#                                     <tgt-dir>/<csv_name>_last100/iter_xxx
# Without suffix: all rows; output goes to <tgt-dir>/<csv_name>/iter_xxx.
# ------------------------------------------------------------
# run_experiment \
#     --gt-csv              "/path/to/gt_a.csv" \
#     --gt-csv              "/path/to/gt_b.csv@@_first_n_=100" \
#     --gt-csv              "/path/to/gt_b.csv@@_last_n_=100" \
#     --gt-csv              "/path/to/gt_c.csv" \
#     --eval-dir            "/path/to/samples_metrics" \
#     --tgt-dir             "/path/to/samples_metrics/asr_results" \
#     --ckpt-start          20000 \
#     --ckpt-end            10000 \
#     --ckpt-interval       2000 \
#     --caption-column      prompt \
#     --index-column        index \
#     --suffix              _0 \
#     --is-extract-ref-text false \
#     --force-rerun         false
