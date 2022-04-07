FROM teaching_image
WORKDIR /app
RUN pip install -r requirements.txt
COPY /modules /app/modules
COPY main.py /app/main.py
CMD ["python3", "main.py"]