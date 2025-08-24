mkdir flask && cd flask

sudo apt update && sudo apt upgrade -y

sudo apt install python3 python3-pip -y

python3 --version

sudo apt install python3-venv -y

python3 -m venv venv

source venv/bin/activate

which python

pip3 list

pip3 freeze > requirements.txt

#can be used for pip3 install -r requirements.txt
