dataset="$1/LSC/SemEval-Latin"
sampling=0
output="$2/LSC/SemEval-Latin"
tokenization_class="StandardTokenization"

python3 "src/tokenization.py" -d "${dataset}" -n "${sampling}" -o "${output}" -t "${tokenization_class}"