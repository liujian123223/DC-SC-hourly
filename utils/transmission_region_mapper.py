# utils/transmission_region_mapper.py

"""
This module maps real-world datacenter location codes to cloud-specific transmission cost regions.

Supported providers: "gcp", "aws", "azure", and "custom"

To define a custom cost matrix:
1. Add your mapping in `location_to_custom_region`
2. Save your transmission cost CSV in: `data/transmission_costs/custom_transmission_cost_matrix.csv`
   - Format: rows and columns must match your custom region names
3. Use `cloud_provider='custom'` when initializing `DatacenterClusterManager`

Each row/col in the CSV must represent the cost per GB from origin -> destination.
"""


# GCP region mapping
location_to_gcp_region = {
    "USA": "us-east1",
    "China": "asia-east1",
    "Japan": "asia-northeast1",
    "France": "europe-west4",
    "India": "asia-south1",
    "Singapore": "asia-southeast1",
    "Canada": "us-east1",
    "Germany": "europe-west1",
    "United_Kingdom": "europe-west1",
    "Australia": "australia-southeast1",
    "Italy": "europe-west1",
    "South_Korea": "asia-northeast1",
    "South_Africa": "africa-south1",
    "Ireland": "europe-west1",
    "UAE": "me-west1",
    "Brazil": "southamerica-east1",
    "Israel": "me-west1",
    "Netherlands": "europe-west1",
    "Spain": "europe-west1",
    "Sweden": "europe-west1",
    "Belgium": "europe-west1",
    "Norway": "europe-west1",
    "Poland": "europe-west1",
    "Switzerland": "europe-west4",
    "US-NY-NYIS": "us-east1",
    "US-CAL-CISO": "us-west1",
    "US-TEX-ERCO": "us-central1",  # Approximation
    "DE-LU": "europe-west1",
    "FR": "europe-west4",
    "SG": "asia-southeast1",
    "JP-TK": "asia-northeast1",
    "IN": "asia-south1",
    "AU-NSW": "australia-southeast1",
    "BR-SP": "southamerica-east1",
    "ZA": "africa-south1",
    "PT": "europe-west1",
    "ES": "europe-west1",
    "BE": "europe-west1",
    "CH": "europe-west4",
    "KR": "asia-northeast1",
    "CA-ON": "us-east1",  # Proxy to closest US East
    "CL-SIC": "southamerica-east1",
    "AT": "europe-west4",
    "NL": "europe-west1"
}

# AWS region mapping
location_to_aws_region = {
    "USA": "us-east-1",
    "China": "ap-southeast-1",
    "Japan": "ap-northeast-1",
    "France": "eu-west-3",
    "India": "ap-south-1",
    "Singapore": "ap-southeast-1",
    "Canada": "ca-central-1",
    "Germany": "eu-central-1",
    "United_Kingdom": "eu-west-2",
    "Australia": "ap-southeast-2",
    "Italy": "eu-south-1",
    "South_Korea": "ap-northeast-2",
    "South_Africa": "af-south-1",
    "Ireland": "eu-west-1",
    "UAE": "ap-southeast-3",
    "Brazil": "sa-east-1",
    "Israel": "eu-central-1",
    "Netherlands": "eu-west-1",
    "Spain": "eu-south-1",
    "Sweden": "eu-central-1",
    "Belgium": "eu-west-1",
    "Norway": "eu-central-1",
    "Poland": "eu-central-1",
    "Switzerland": "eu-central-2",
    "US-NY-NYIS": "us-east-1",
    "US-CAL-CISO": "us-west-1",
    "US-TEX-ERCO": "us-east-1-dwf-1",
    "DE-LU": "eu-central-1",
    "FR": "eu-west-3",
    "SG": "ap-southeast-1",
    "JP-TK": "ap-northeast-1",
    "IN": "ap-south-1",
    "IN-WE": "ap-south-1",
    "AU-NSW": "ap-southeast-2",
    "AU-VIC": "ap-southeast-2",
    "BR-SP": "sa-east-1",
    "ZA": "af-south-1",
    "PT": "eu-south-1",
    "ES": "eu-south-1",
    "BE": "eu-west-1",
    "CH": "eu-central-1",
    "KR": "ap-northeast-2",
    "CA-ON": "ca-central-1",
    "CL-SIC": "us-east-1-chl-1",
    "US-MIDA-PJM": "us-east-1-dwf-1",
    "AT": "eu-central-1",
    "NL": "eu-west-1"
}

# AZURE region mapping
location_to_azure_region = {
    "USA": "East US",
    "China": "Southeast Asia",
    "Japan": "Japan East",
    "France": "France Central",
    "India": "Central India",
    "Singapore": "Southeast Asia",
    "Canada": "Canada Central",
    "Germany": "Germany West Central",
    "United_Kingdom": "North Europe",
    "Australia": "Australia East",
    "Italy": "West Europe",
    "South_Korea": "Korea Central",
    "South_Africa": "South Africa North",
    "Ireland": "North Europe",
    "UAE": "Southeast Asia",
    "Brazil": "Brazil South",
    "Israel": "West Europe",
    "Netherlands": "West Europe",
    "Spain": "Spain Central",
    "Sweden": "North Europe",
    "Belgium": "West Europe",
    "Norway": "North Europe",
    "Poland": "Germany West Central",
    "Switzerland": "Switzerland North",
    "US-NY-NYIS": "East US",
    "US-CAL-CISO": "West US",
    "US-TEX-ERCO": "South Central US",
    "DE-LU": "Germany West Central",
    "FR": "France Central",
    "SG": "Southeast Asia",
    "JP-TK": "Japan East",
    "IN": "Central India",
    "AU-NSW": "Australia East",
    "BR-SP": "Brazil South",
    "ZA": "South Africa North",
    "PT": "Portugal North",
    "ES": "Spain Central",
    "BE": "West Europe",
    "CH": "Switzerland North",
    "KR": "Korea Central",
    "CA-ON": "Canada Central",
    "CL-SIC": "Chile North",
    "AT": "Austria East",
    "NL": "North Europe"
}

# Custom region mapping
location_to_custom_region = {
    "USA": "CustomRegion1",
    "China": "CustomRegion2",
    "Japan": "CustomRegion7",
    "France": "CustomRegion5",
    "India": "CustomRegion8",
    "Singapore": "CustomRegion6",
    "Canada": "CustomRegion1",
    "Germany": "CustomRegion4",
    "United_Kingdom": "CustomRegion5",
    "Australia": "CustomRegion9",
    "Italy": "CustomRegion5",
    "South_Korea": "CustomRegion7",
    "South_Africa": "CustomRegion11",
    "Ireland": "CustomRegion5",
    "UAE": "CustomRegion6",
    "Brazil": "CustomRegion10",
    "Israel": "CustomRegion5",
    "Netherlands": "CustomRegion5",
    "Spain": "CustomRegion5",
    "Sweden": "CustomRegion5",
    "Belgium": "CustomRegion5",
    "Norway": "CustomRegion5",
    "Poland": "CustomRegion4",
    "Switzerland": "CustomRegion5",
    "US-NY-NYIS": "CustomRegion1",
    "US-CAL-CISO": "CustomRegion2",
    "US-TEX-ERCO": "CustomRegion3",
    "DE-LU": "CustomRegion4",
    "FR": "CustomRegion5",
    "SG": "CustomRegion6",
    "JP-TK": "CustomRegion7",
    "IN": "CustomRegion8",
    "AU-NSW": "CustomRegion9",
    "BR-SP": "CustomRegion10",
    "ZA": "CustomRegion11",
    # Add more mappings as needed
}

import warnings

def map_location_to_region(location_code: str, provider: str):
    provider = provider.lower()
    if provider == "gcp":
        region_map = location_to_gcp_region
    elif provider == "aws":
        region_map = location_to_aws_region
    elif provider == "azure":
        region_map = location_to_azure_region
    elif provider == "custom":
        region_map = location_to_custom_region
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use one of: gcp, aws, azure, custom.")

    # === Exact match ===
    region = region_map.get(location_code)
    if region:
        return region

    # === Fallback fuzzy match ===
    for key in region_map:
        if key in location_code or location_code in key:
            warnings.warn(
                f"[map_location_to_region] WARNING: No exact match for '{location_code}', "
                f"using closest match: '{key}' -> {region_map[key]}"
            )
            return region_map[key]

    # === Nothing found ===
    raise ValueError(f"[map_location_to_region] ERROR: Could not map location '{location_code}' to any known region for provider '{provider}'.")

