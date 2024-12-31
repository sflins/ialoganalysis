clear
export MONGODB_URL='mongodb+srv://loganalysis:Dn750102@loganalysismongodb.qvevx.mongodb.net'
export MONGODB_DBNAME='logs_database'

pip install -r requirements.txt

sudo snap install ollama
#pip install ollama

ollama pull phi
ollama pull llama2
ollama pull codellama

sudo snap restart ollama.listener

python3 app.py
