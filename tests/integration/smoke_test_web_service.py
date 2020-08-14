import time
from argparse import ArgumentParser

import requests

input_sample = [
    {
        "age": 50,
        "gender": "female",
        "systolic": 110,
        "diastolic": 80,
        "height": 175,
        "weight": 80,
        "cholesterol": "normal",
        "glucose": "normal",
        "smoker": "not-smoker",
        "alcoholic": "not-alcoholic",
        "active": "active",
    }
]

output_sample = {"probability": [0.26883566156891225]}


def main():
    retries = 100

    for i in range(retries):
        try:
            # Get service uri and payload
            scoring_uri = parse_args()
            request_payload = {"data": input_sample}

            # Make request
            response = requests.post(scoring_uri, json=request_payload)
            response_payload = response.json()

            # Should return valid response payload
            assert response.status_code == 200
            assert "probability" in response_payload.keys()
            assert type(response_payload["probability"]) == list
            assert type(response_payload["probability"][0]) == float

            return response_payload

        except requests.exceptions.HTTPError as e:
            if i == retries - 1:
                raise e
            print("Retrying...")
            print(e)
            time.sleep(1)


def parse_args():
    # Parse command line arguments
    ap = ArgumentParser()
    ap.add_argument("--scoring-uri", required=True)
    args = vars(ap.parse_args())
    return args["scoring_uri"]


if __name__ == "__main__":
    main()
