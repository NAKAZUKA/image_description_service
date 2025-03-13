import requests, logging, os

from PIL import Image, UnidentifiedImageError


def load_image(image_url: str = None, local_path: str = None) -> Image.Image:
    try:
        if local_path:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"File not found: {local_path}")
            return Image.open(local_path)
        elif image_url:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw)
        else:
            raise ValueError("Either image_url or local_path must be provided.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching image: {str(e)}")
        raise ValueError(f"Failed to load image from URL: {str(e)}")
    except UnidentifiedImageError:
        logging.error("Invalid image format or corrupted file.")
        raise ValueError("Invalid image format or corrupted file.")