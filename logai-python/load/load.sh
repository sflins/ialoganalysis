clear
export MONGODB_URL='mongodb+srv://loganalysis:Dn750102@loganalysismongodb.qvevx.mongodb.net'
export MONGODB_DBNAME='logs_database'

export GITREPO_TOKEN='
export GITREPO_OWNER='sflins'
export GITREPO_NAME='loganalysis'
export GITREPO_BRANCH='main'

export LOGS_DIR='/home/ubuntu/projects/loganalysis/logai-python/store/logs'
export SPRINGBOOT_GUIDE_PDF='/home/ubuntu/projects/loganalysis/logai-python/store/docs/spring-boot-reference.pdf'


python3 install -r requiriments.txt

python3 data-pipeline.py

