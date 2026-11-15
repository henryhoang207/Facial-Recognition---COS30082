"""Start the web app from the repo root:  python -m app.run  [--config app/config.yaml]"""

import argparse

import uvicorn

from app.backend.main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="FaceGuard AI web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=None, help="default: app/config.yaml")
    args = parser.parse_args()
    uvicorn.run(create_app(args.config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
