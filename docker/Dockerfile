FROM python:3.11

# Set the timezone
ENV TZ=Asia/Bangkok

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy requirements.txt and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
