# PYTHON API SAMEPLE

#### How to run

```
./run.sh
```

#### How to run module pipeline with torch:

Download model, password: `intel123`
```bash
scp -r ziniu@lic-code-vm13:/home/ziniu/web_files/models/Qwen2.5-VL-3B-Instruct/torch ../../cpp/module_genai/ut_pipelines/Qwen2.5-VL-3B-Instruct/
```
Run test:

```bash
source ../../../../python-env/bin/activate
pip install -r requirements.txt
bash run_pipeline_with_torch.sh
```