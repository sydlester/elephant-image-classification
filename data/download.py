import aiohttp
import asyncio
import pandas as pd
from azure.storage.blob import BlobServiceClient
import nest_asyncio
from dotenv import load_dotenv
import os

load_dotenv()
nest_asyncio.apply()

# Configuration
# CONCURRENT_DOWNLOADS = 50
TIMEOUT = aiohttp.ClientTimeout(total=30)
MAX_RETRIES = 3
CONTAINER_NAME = os.environ.get('CONTAINER')
AZURE_CONNECTION_STRING = os.environ.get('AZURE_STRING')

consensus_data = pd.read_csv('consensus_data.csv')[['CaptureEventID']]
images = pd.read_csv('all_images.csv')

# Initialize Azure Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
blob_container_client = blob_service_client.get_container_client(CONTAINER_NAME)

FAILED_IMAGES = []

async def download_image(session, url, timeout=TIMEOUT):
    full_url = "https://snapshotserengeti.s3.msi.umn.edu/" + url
    retries = 0
    while retries < MAX_RETRIES:
        try:
            async with session.get(full_url, timeout=timeout) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    print(f"Failed to download {url}: HTTP {response.status}")
                    return None
        except aiohttp.ClientError as e:
            print(f"Client error: {str(e)}, downloading {url}")
        except asyncio.TimeoutError:
            print(f"Timeout error downloading {url}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}, downloading {url}")
        
        retries += 1
        print(f"Retrying ({retries}/{MAX_RETRIES})...")
        await asyncio.sleep(5)  # Sleep for 5 seconds before retrying

    print(f"Failed to download {url} after {MAX_RETRIES} retries.")
    FAILED_IMAGES.append(url)
    return None

async def upload_to_blob(blob_name, data):
    try:
        blob_client = blob_container_client.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=True)
    except Exception as e:
        print(f"Error uploading {blob_name}: {e}")

async def process_image(session, url, idx, timeout=TIMEOUT):
    image_data = await download_image(session, url, timeout)
    if image_data:
        blob_name = f"{url}"
        await upload_to_blob(blob_name, image_data)

async def download_and_upload_images(url_list, timeout=TIMEOUT):
    async with aiohttp.ClientSession() as session:
        tasks = [process_image(session, url, idx, timeout) for idx, url in enumerate(url_list)]
        await asyncio.gather(*tasks)

async def main():
    # Read URLs from CSV
    df = pd.merge(images, consensus_data, on='CaptureEventID')
    urls = df['URL_Info'].tolist()

    # Split the URLs into batches to avoid memory overload
    batch_size = 1000  # Number of images to download per batch
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        await download_and_upload_images(batch)
        print(f"Batch {i // batch_size + 1} complete.")

    print(f'Failed downloads: {FAILED_IMAGES} \nRetrying...')

    # retry failed downloads with a higher timeout limit
    batch_size = max(len(FAILED_IMAGES), 1000)
    for i in range(0, len(FAILED_IMAGES), batch_size):
        batch = FAILED_IMAGES[i:i + batch_size]
        await download_and_upload_images(batch, timeout=aiohttp.ClientTimeout(total=60))
        print(f"Batch {i // batch_size + 1} complete.")

if __name__ == "__main__":
    asyncio.run(main())
