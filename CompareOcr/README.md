# activate virtual env
source ./compare_ocr_env/bin/activate

# start endpoint
python3 start_api_server.py

# call endpoint
curl http://127.0.0.1:8000/

# serve the .html file
python3 -m http.server 8888

port forward 8888

