FROM teaching-base
WORKDIR /app
COPY /modules /app/modules
COPY main.py /app/main.py
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
RUN rm requirements.txt
CMD ["python3", "main.py"]