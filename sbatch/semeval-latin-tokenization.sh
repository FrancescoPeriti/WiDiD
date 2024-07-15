#dataset="$1/LSC/SemEval-Latin"
#sampling=0
#output="$2/LSC/SemEval-Latin"
#tokenization_class="StandardTokenization"
#python3 "src/tokenization.py" -d "${dataset}" -n "${sampling}" -o "${output}" -t "${tokenization_class}"

dataset="$1/LSC/SemEval-Latin"
sampling=0
output="$2/LSC/SemEval-Latin"
tokenization_class="LatinSpacyTokenization"
model='la_core_web_lg'

pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-any-py3-none-any.whl
python3 "src/tokenization.py" -d "${dataset}" -n "${sampling}" -o "${output}" -t "${tokenization_class}" -m "${model}"
