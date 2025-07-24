# activate virtual env
source ./compare_ocr_env/bin/activate

# start endpoint
fastapi dev main.py

# call endpoint
curl http://127.0.0.1:8000/
