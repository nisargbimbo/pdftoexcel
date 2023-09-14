FROM python:3.7

WORKDIR /app

#COPY packages.txt ./packages.txt
RUN apt-get update
RUN apt-get install -y libgl1
RUN apt-get install -y poppler-utils

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8080

COPY . .

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py", "--server.port", "8080"]