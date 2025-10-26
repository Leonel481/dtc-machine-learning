FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

COPY pyproject.toml .
RUN pip install --no-cache-dir .


COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]