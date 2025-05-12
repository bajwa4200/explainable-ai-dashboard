# Explainable AI Dashboard

Train a random forest on the Iris dataset, compute permutation feature importances, and browse the results in a small Flask dashboard with a JSON API.

## Quick start

```bash
pip install -e ".[dev]"
xai-dashboard
```

Open http://127.0.0.1:5050/ for the HTML table or `/api/metrics` for JSON.

## Tests

```bash
python -m pytest -q
```

## Layout

```
explainable-ai-dashboard/
├── xai/
│   ├── model.py
│   ├── dashboard.py
│   └── cli.py
└── tests/
```

## License

MIT — see [LICENSE](LICENSE).
