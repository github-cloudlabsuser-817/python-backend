###
python -m venv myenv
myenv\scripts\activate
pip install -r requirements.txt
uvicorn server:app --reload
###

# docker local run
# docker build -t my-python-app .
# docker run my-python-app


# uvicorn docker local run
# docker build -t my-fastapi-app .
# docker run -p 8000:8000 my-fastapi-app

netstat -aon | findstr :8080    

docker build -t my-fastapi-app:v1.2 .
docker run -p 8000:8000 my-fastapi-app:v1.2

docker build -t my-fastapi-app:v1.4 .
docker run -p 5002:5002 my-fastapi-app:v1.4 .