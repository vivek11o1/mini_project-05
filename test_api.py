"""Smoke test for the PriceNest FastAPI service. Start it with: uvicorn api:app --reload"""

import json
import urllib.error
import urllib.request

BASE_URL = "http://localhost:8000"


def request(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.load(response)


def main() -> None:
    info = request("/")
    assert info["docs"] == "/docs", info

    health = request("/health")
    assert health["status"] == "ok" and health["model_loaded"], health

    single = request("/predict", {"area": 1500, "bedrooms": 3, "bathrooms": 2})
    assert "predicted_price" in single, single
    print("Single prediction:", json.dumps(single, indent=2))

    batch = request(
        "/predict/batch",
        {
            "houses": [
                {"area": 1200, "bedrooms": 2, "bathrooms": 1},
                {"area": 2500, "bedrooms": 4, "bathrooms": 3},
            ]
        },
    )
    assert len(batch["predictions"]) == 2, batch
    print("Batch prices:", [p["predicted_price"] for p in batch["predictions"]])

    print("\nAll API smoke tests passed. Swagger UI:", BASE_URL + "/docs")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as error:
        raise SystemExit(f"Could not reach the API. Start it with 'uvicorn api:app --reload'. {error}")
