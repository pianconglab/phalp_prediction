FROM python:3.7.4-slim

WORKDIR /phalp_prediction

# 复制文件
COPY . .

# 安装依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# 暴露应用端口
EXPOSE 8080

# 运行应用程序
ENTRYPOINT ["python3", "src/run.py"]

# 设置健康检查
HEALTHCHECK --interval=30s \
            --timeout=5s \
            --start-period=120s \
            --retries=3 \
            CMD curl --silent --fail http://localhost:8080/health || exit 1