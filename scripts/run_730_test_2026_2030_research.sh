#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ORACLE_SCRIPT="${ORACLE_SCRIPT:-scripts/7.30-test.py}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-results/exp_730_test_2026_2030_research}"
LOG_DIR="${BASE_OUTPUT_DIR}/logs"
FAILED_CASES_PATH="${BASE_OUTPUT_DIR}/failed_cases.tsv"
COMMANDS_PATH="${BASE_OUTPUT_DIR}/commands.tsv"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"

# core: Fig1-Fig4 paper-oriented sweeps.
# full_factorial: all 5 years x 3 carbon policies x 4 PUE scenarios x 3 deployment matrices.
EXPERIMENT_SET="${EXPERIMENT_SET:-core}"

HORIZON_HOURS="${HORIZON_HOURS:-8760}"
LLM_TRAINING_RATIO="${LLM_TRAINING_RATIO:-0.3}"
IT_CAPACITY_UTILIZATION="${IT_CAPACITY_UTILIZATION:-1.0}"
IT_CAPACITY_MULTIPLIER="${IT_CAPACITY_MULTIPLIER:-1.0}"
HIGH_CARBON_QUANTILE="${HIGH_CARBON_QUANTILE:-0.75}"
OPTION_WORKERS="${OPTION_WORKERS:-0}"
OPTION_CHUNK_SIZE="${OPTION_CHUNK_SIZE:-0}"
MAX_TASKS="${MAX_TASKS:-}"
WRITE_UTILIZATION="${WRITE_UTILIZATION:-0}"

YEARS=(2026 2027 2028 2029 2030)
CARBON_POLICIES=(CP NDC NZ) 
PUE_CASES=(
  "Base|Base"
  "Lift_Off|Lift-Off"
  "High_Efficiency|High Efficiency"
  "Headwinds|Headwinds"
)
MATRIX_CASES=(
  "UGD_global_transfer|global_transfer|data/network_cost/policy_scenarios/azure_global_transfer_with_safeguards.csv"
  "BCD_eea_trusted|eea_trusted|data/network_cost/policy_scenarios/azure_eea_adequacy_trusted_zone.csv"
  "SDS_eea_only|eea_only|data/network_cost/policy_scenarios/azure_eea_only_regional_residency.csv"
)
PRACTICE_CASES=(
  "best_practice|NZ|UGD_global_transfer|global_transfer|data/network_cost/policy_scenarios/azure_global_transfer_with_safeguards.csv"
  "worst_practice|CP|SDS_eea_only|eea_only|data/network_cost/policy_scenarios/azure_eea_only_regional_residency.csv"
)

mkdir -p "${LOG_DIR}"
printf 'experiment_group\tyear\tcarbon_policy\tpue_scenario\tdeployment_label\tmatrix_label\texit_code\tlog_path\n' > "${FAILED_CASES_PATH}"
printf 'experiment_group\tyear\tcarbon_policy\tpue_scenario\tdeployment_label\tmatrix_label\toutput_dir\tlog_path\tcommand\n' > "${COMMANDS_PATH}"

run_case() {
  local experiment_group="$1"
  local year="$2"
  local carbon_policy="$3"
  local pue_label="$4"
  local pue_scenario="$5"
  local deployment_label="$6"
  local matrix_label="$7"
  local matrix_path="$8"

  local output_dir="${BASE_OUTPUT_DIR}/${experiment_group}/${year}/${carbon_policy}_${pue_label}_${deployment_label}"
  local log_path="${LOG_DIR}/${experiment_group}_${year}_${carbon_policy}_${pue_label}_${deployment_label}.log"
  mkdir -p "${output_dir}"

  local cmd=(
    "${PYTHON_BIN}" "${ORACLE_SCRIPT}"
    --year "${year}"
    --run-baselines
    --solve mcf
    --carbon-data-root data/EM-estimate
    --carbon-policy "${carbon_policy}"
    --pue-scenario "${pue_scenario}"
    --llm-training-ratio "${LLM_TRAINING_RATIO}"
    --cloud-provider azure
    --policy-transmission-matrix "${matrix_path}"
    --it-capacity-utilization "${IT_CAPACITY_UTILIZATION}"
    --it-capacity-multiplier "${IT_CAPACITY_MULTIPLIER}"
    --high-carbon-quantile "${HIGH_CARBON_QUANTILE}"
    --option-workers "${OPTION_WORKERS}"
    --option-chunk-size "${OPTION_CHUNK_SIZE}"
    --horizon-hours "${HORIZON_HOURS}"
    --output-dir "${output_dir}"
    --greedy-infeasible-fallback constrained
  )
  if [[ -n "${MAX_TASKS}" ]]; then
    cmd+=(--max-tasks "${MAX_TASKS}")
  fi
  if [[ "${WRITE_UTILIZATION}" == "1" ]]; then
    cmd+=(--write-utilization)
  fi

  local quoted_cmd
  printf -v quoted_cmd '%q ' "${cmd[@]}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${experiment_group}" "${year}" "${carbon_policy}" "${pue_scenario}" \
    "${deployment_label}" "${matrix_label}" "${output_dir}" "${log_path}" "${quoted_cmd% }" \
    >> "${COMMANDS_PATH}"

  echo "============================================================"
  echo "Starting case: group=${experiment_group}, year=${year}, policy=${carbon_policy}, pue=${pue_scenario}, deployment=${deployment_label}"
  echo "Output dir:    ${output_dir}"
  echo "Log path:      ${log_path}"
  echo "Started at:    $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  if "${cmd[@]}" > "${log_path}" 2>&1; then
    echo "Finished case: group=${experiment_group}, year=${year}, policy=${carbon_policy}, pue=${pue_scenario}, deployment=${deployment_label}"
    echo "Finished at:   $(date '+%Y-%m-%d %H:%M:%S')"
  else
    local exit_code="$?"
    echo "Failed case: group=${experiment_group}, year=${year}, policy=${carbon_policy}, pue=${pue_scenario}, deployment=${deployment_label}, exit_code=${exit_code}"
    echo "Check log: ${log_path}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${experiment_group}" "${year}" "${carbon_policy}" "${pue_scenario}" \
      "${deployment_label}" "${matrix_label}" "${exit_code}" "${log_path}" \
      >> "${FAILED_CASES_PATH}"
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      return "${exit_code}"
    fi
  fi
}

run_core_experiments() {
  local matrix_case pue_case year carbon_policy
  local pue_label pue_scenario deployment_label matrix_label matrix_path

  IFS='|' read -r deployment_label matrix_label matrix_path <<< "${MATRIX_CASES[0]}"
  carbon_policy="CP"
  for year in "${YEARS[@]}"; do
    for pue_case in "${PUE_CASES[@]}"; do
      IFS='|' read -r pue_label pue_scenario <<< "${pue_case}"
      run_case "pue_capacity_sweep" "${year}" "${carbon_policy}" "${pue_label}" "${pue_scenario}" "${deployment_label}" "${matrix_label}" "${matrix_path}"
    done
  done

  IFS='|' read -r pue_label pue_scenario <<< "${PUE_CASES[0]}"
  IFS='|' read -r deployment_label matrix_label matrix_path <<< "${MATRIX_CASES[0]}"
  for year in "${YEARS[@]}"; do
    for carbon_policy in "${CARBON_POLICIES[@]}"; do
      run_case "carbon_policy_sweep" "${year}" "${carbon_policy}" "${pue_label}" "${pue_scenario}" "${deployment_label}" "${matrix_label}" "${matrix_path}"
    done
  done

  carbon_policy="CP"
  IFS='|' read -r pue_label pue_scenario <<< "${PUE_CASES[0]}"
  for year in "${YEARS[@]}"; do
    for matrix_case in "${MATRIX_CASES[@]}"; do
      IFS='|' read -r deployment_label matrix_label matrix_path <<< "${matrix_case}"
      run_case "deployment_governance_sweep" "${year}" "${carbon_policy}" "${pue_label}" "${pue_scenario}" "${deployment_label}" "${matrix_label}" "${matrix_path}"
    done
  done

  for year in "${YEARS[@]}"; do
    for pue_case in "${PUE_CASES[@]}"; do
      IFS='|' read -r pue_label pue_scenario <<< "${pue_case}"
      for practice_case in "${PRACTICE_CASES[@]}"; do
        IFS='|' read -r practice_label carbon_policy deployment_label matrix_label matrix_path <<< "${practice_case}"
        run_case "integrated_path_${practice_label}" "${year}" "${carbon_policy}" "${pue_label}" "${pue_scenario}" "${deployment_label}" "${matrix_label}" "${matrix_path}"
      done
    done
  done
}

run_full_factorial_experiments() {
  local matrix_case pue_case year carbon_policy
  local pue_label pue_scenario deployment_label matrix_label matrix_path

  for year in "${YEARS[@]}"; do
    for carbon_policy in "${CARBON_POLICIES[@]}"; do
      for pue_case in "${PUE_CASES[@]}"; do
        IFS='|' read -r pue_label pue_scenario <<< "${pue_case}"
        for matrix_case in "${MATRIX_CASES[@]}"; do
          IFS='|' read -r deployment_label matrix_label matrix_path <<< "${matrix_case}"
          run_case "full_factorial" "${year}" "${carbon_policy}" "${pue_label}" "${pue_scenario}" "${deployment_label}" "${matrix_label}" "${matrix_path}"
        done
      done
    done
  done
}

case "${EXPERIMENT_SET}" in
  core)
    total=$(( ${#YEARS[@]} * (${#PUE_CASES[@]} + ${#CARBON_POLICIES[@]} + ${#MATRIX_CASES[@]} + ${#PUE_CASES[@]} * ${#PRACTICE_CASES[@]}) ))
    echo "Experiment set: core Fig1-Fig4 paper sweeps"
    echo "Total cases: ${total}"
    run_core_experiments
    ;;
  full_factorial)
    total=$(( ${#YEARS[@]} * ${#CARBON_POLICIES[@]} * ${#PUE_CASES[@]} * ${#MATRIX_CASES[@]} ))
    echo "Experiment set: full factorial"
    echo "Total cases: ${total}"
    run_full_factorial_experiments
    ;;
  *)
    echo "Unknown EXPERIMENT_SET=${EXPERIMENT_SET}. Use core or full_factorial." >&2
    exit 2
    ;;
esac

echo "All ${total} cases requested."
echo "Results:  ${BASE_OUTPUT_DIR}"
echo "Commands: ${COMMANDS_PATH}"
echo "Failures: ${FAILED_CASES_PATH}"
echo "Logs:     ${LOG_DIR}"
