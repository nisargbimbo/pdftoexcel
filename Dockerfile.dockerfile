FROM python:3.7

WORKDIR /app

#COPY packages.txt ./packages.txt
RUN apt-get update
RUN apt-get install -y libgl1
RUN apt-get install -y poppler-utils
COPY libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb ./libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
RUN apt-get install ./libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8080

COPY . .

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py", "--server.port", "8080"]