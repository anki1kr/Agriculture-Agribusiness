# Ankit Kumar | Junior ML Data Analyst
# Week 2: USDA NASS API Collector & Rate Limiting

import os
import time
import requests
import pandas as pd

def fetch_usda_data(api_key=None, state="IA", year="2023", mock=True):
    api_key = api_key or os.environ.get("USDA_NASS_API_KEY")
    endpoint = "https://quickstats.nass.usda.gov/api/api_GET/"
    params = {
        "key": api_key or "",
        "source_desc": "SURVEY",
        "sector_desc": "CROPS",
        "group_desc": "FIELD CROPS",
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "state_alpha": state,
        "year": str(year),
        "format": "JSON"
    }

    # live query
    if api_key:
        print(f"[*] Querying USDA NASS: state={state}, yr={year}...")
        for attempt in range(5):
            try:
                res = requests.get(endpoint, params=params, timeout=25)
                if res.status_code == 200:
                    data = res.json().get("data", [])
                    print(f"[+] Pulled {len(data):,} rows from USDA.")
                    return pd.DataFrame(data)
                elif res.status_code == 429:
                    wait = 2 ** attempt
                    print(f"[!] Hit 429 rate limit, sleeping {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[!] Status {res.status_code}, aborting attempt.")
                    break
            except requests.RequestException as e:
                print(f"[!] Network error: {e}")
                time.sleep(2)

    # offline benchmark cache
    if mock:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_path = os.path.join(base_dir, "data", "crop_yield_district_panel.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            subset = df[df["crop_year"] == int(year)].copy()
            print(f"[+] Loaded {len(subset):,} records from local panel for year {year}")
            return subset

    raise RuntimeError("Failed to fetch USDA data and no local cache found.")

if __name__ == "__main__":
    df = fetch_usda_data(year="2023")
    print(df[["district_id", "crop_year", "yield_tha", "harvested_ha"]].head())
