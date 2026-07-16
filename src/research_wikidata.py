import requests
import urllib.parse

def test_wikidata():
    # Query all items located in Iraq (Q796) that have an image (P18) and coordinates (P625)
    query = """
    SELECT ?item ?itemLabel ?coord ?image WHERE {
      ?item wdt:P17 wd:Q796;
            wdt:P625 ?coord;
            wdt:P18 ?image.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    """
    
    url = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/json",
        "User-Agent": "Antigravity/1.0 (Contact: antigravity@example.com)"
    }
    params = {"query": query}
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", {}).get("bindings", [])
        print(f"Found {len(results)} locations with images in Iraq from Wikidata!")
        for idx, r in enumerate(results[:5]):
            print(f"- {r['itemLabel']['value']}: {r['coord']['value']}")
    else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_wikidata()
