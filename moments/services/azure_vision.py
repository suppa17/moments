import os
import requests

from dotenv import load_dotenv

# Load the .env file
load_dotenv()


# Read these from environment variables (or a separate config file).
AZURE_VISION_ENDPOINT = os.getenv("AZURE_VISION_ENDPOINT")  # e.g., "https://YOUR_REGION.api.cognitive.microsoft.com"
AZURE_VISION_KEY =  os.getenv("AZURE_VISION_KEY")            # your subscription key


def generate_alt_text(image_path: str) -> str:
    """
    Call Azure Computer Vision's "Describe Image" endpoint to generate alt text for an image.
    Returns the best caption from the response or a fallback string if no caption is found.
    """
    if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
        # If keys are missing, log a warning or return a fallback
        return "No alt text (Azure credentials not set)."

    # Example endpoint for "Describe" feature in API version 3.2:
    vision_url = f"{AZURE_VISION_ENDPOINT}/vision/v3.2/analyze?visualFeatures=Description"

    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_VISION_KEY,
        'Content-Type': 'application/octet-stream'
    }
    params = {
        'maxCandidates': '1',  
        'language': 'en'
    }

    # Read the local image file into memory
    with open(image_path, 'rb') as f:
        image_data = f.read()

    # Make the POST request to Azure
    response = requests.post(vision_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()  # raise an exception for HTTP errors

    result = response.json()
    captions = result.get("description", {}).get("captions", [])
    if captions:
        # Return the text from the top (best) caption
        return captions[0]["text"]
    else:
        return "No alt text (no caption returned)."


def generate_image_tags(image_path: str) -> list[str]:
    """
    Calls the Azure Computer Vision 'analyze' endpoint to identify objects/tags in the image.
    Returns a list of tags (strings).
    """
    if not AZURE_VISION_ENDPOINT or not AZURE_VISION_KEY:
        return []  # return an empty list if credentials are missing

    # 'visualFeatures=Tags' to let Azure guess keywords for the image
    vision_url = f"{AZURE_VISION_ENDPOINT}/vision/v3.2/analyze?visualFeatures=Tags"
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_VISION_KEY,
        'Content-Type': 'application/octet-stream'
    }
    params = {
        'visualFeatures': 'Tags',
        'language': 'en'
    }

    with open(image_path, 'rb') as f:
        image_data = f.read()

    response = requests.post(vision_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()
    result = response.json()

    # result["tags"] is a list of dicts, e.g. {"name": "cat", "confidence": 0.9}
    tags = []
    for tag_info in result.get("tags", []):
        tag_name = tag_info.get("name")
        confidence = tag_info.get("confidence", 0)
        # Optionally filter out low-confidence tags
        if confidence >= 0.5:
            tags.append(tag_name)
    return tags
