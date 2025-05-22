import os
import argparse
from uvicorn import run

def parse_args():
    parser = argparse.ArgumentParser("Protein Embedding HTTP Server")
    parser.add_argument("-l", "--listen", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("-p", "--port", type=int, default=8080, help="监听端口")
    parser.add_argument("-w", "--workers", type=int, default=2,
                        help="SeqVecEmbedder 实例数 (环境变量 WORKERS)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 设置环境变量，供 lifespan 读取
    os.environ["WORKERS"] = str(args.workers)
    run("phalp_prediction:app", host=args.listen, port=args.port)