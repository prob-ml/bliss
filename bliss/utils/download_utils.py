import base64
from typing import Dict, Optional

import requests


def download_git_lfs_file(url, headers: Optional[Dict[str, str]] = None) -> bytes:
    """Download a file from git-lfs.

    Args:
        url (str): URL to git-lfs file.
        headers (dict, optional): Headers to pass to requests.get. Defaults to None.

    Returns:
        bytes: File contents.
    """
    ptr_file = requests.get(url, headers=headers, timeout=10)
    ptr = ptr_file.json()
    ptr_sha = ptr["sha"]

    blob_file = requests.get(
        f"https://api.github.com/repos/prob-ml/bliss/git/blobs/{ptr_sha}",
        headers=headers,
        timeout=10,
    )
    blob = blob_file.json()
    blob_content = blob["content"]
    assert blob["encoding"] == "base64"

    blob_decoded = base64.b64decode(blob_content).decode("utf-8").split("\n")
    sha = blob_decoded[1].split(" ")[1].split(":")[1]
    size = int(blob_decoded[2].split(" ")[1])

    lfs_req_headers = {
        "Accept": "application/vnd.git-lfs+json",
        # Already added when you pass json=
        # 'Content-type': 'application/json',
    }
    if headers:
        lfs_req_headers.update(headers)
    lfs_ptr_file = requests.post(
        "https://github.com/prob-ml/bliss.git/info/lfs/objects/batch",
        headers=lfs_req_headers,
        json={
            "operation": "download",
            "transfer": ["basic"],
            "objects": [
                {
                    "oid": sha,
                    "size": size,
                }
            ],
        },
        timeout=10,
    )
    lfs_ptr = lfs_ptr_file.json()
    lfs_ptr_download_url = lfs_ptr["objects"][0]["actions"]["download"]["href"]  # noqa: WPS219

    # Get and write weights to pretrained weights path
    file = requests.get(lfs_ptr_download_url, timeout=10)
    return file.content
