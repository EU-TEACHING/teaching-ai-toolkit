FROM teaching-base
ARG ARCH
WORKDIR /app
COPY /modules /app/modules
COPY main.py /app/main.py

RUN if [ "${ARCH}" = "x86" ]; then \
        python3 -m pip install tensorflow===2.8.0; \
    elif [ "${ARCH}" = "arm" ]; then \
        python3 -m pip install tensorflow -f https://tf.kmtea.eu/whl/stable.html; \
    fi;
RUN python3 -m pip install tensorflow-addons

CMD ["python3", "main.py"]