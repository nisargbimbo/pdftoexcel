FROM python:3.7

WORKDIR /app

#COPY packages.txt ./packages.txt
# RUN apt-get install libgl1
# RUN aptget install poppler-utils

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8080

COPY . .

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py", "--server.port", "8080"]