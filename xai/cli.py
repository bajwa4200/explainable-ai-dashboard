"""Run the Flask dashboard."""

from __future__ import annotations


def main() -> None:
    from xai.dashboard import create_app

    app = create_app()
    app.run(host="127.0.0.1", port=5050, debug=False)


if __name__ == "__main__":
    main()
