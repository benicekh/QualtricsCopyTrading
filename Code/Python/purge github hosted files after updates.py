import requests

# CDN and file base
cdn_base = "https://cdn.jsdelivr.net/gh/benicekh/QualtricsCopyTrading@main/"
static_files = ["Code/JS/copyTraining.js"]

suffixes = [
    None,
    'no_risk_start_bottom_high',
    'no_risk_start_bottom_low',
    'no_risk_start_top_high',
    'no_risk_start_top_low',
    'risk_start_bottom_high',
    'risk_start_bottom_low',
    'risk_start_top_high',
    'risk_start_top_low'
]

# Suffix-based files
dynamic_files = [
    "Paths/javastrings_price_series_{}.js",
    "CRRAPaths/javastrings_price_series_{}.js",
    "Botdata/javastrings_bots_{}.js",
    "TLdata/javastrings_TLs_series_{}.js"
]

purged_urls = []

# Construct URLs to purge
for suffix in suffixes:
    if suffix is None:
        continue
    for path in dynamic_files:
        purged_urls.append(cdn_base + path.format(suffix))

# Add static file
purged_urls.extend(cdn_base + f for f in static_files)

# Check if file exists before purge
def file_exists(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except Exception as e:
        print(f"[!] Error checking existence of {url}: {e}")
        return False

# Purge function
def purge_url(url):
    purge_path = url.split("@main/")[1]
    api_url = f"https://purge.jsdelivr.net/gh/benicekh/QualtricsCopyTrading@main/{purge_path}"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            print(f"[✓] Purged: {url}")
        else:
            print(f"[✗] Failed to purge: {url} — {response.status_code} — {response.text}")
    except Exception as e:
        print(f"[!] Exception during purge: {e}")

# Execute with existence check
for url in purged_urls:
    if file_exists(url):
        print(f"[i] Found: {url}")
        purge_url(url)
    else:
        print(f"[×] Skipped (not found on CDN): {url}")