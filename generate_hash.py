import hashlib

# The name of the model file
file_name = "yolov8n.pt"

# Open the file in binary read mode
with open(file_name, "rb") as f:
    # Read the contents of the file
    file_bytes = f.read()
    # Calculate the SHA256 hash
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()

print(f"The SHA256 hash of '{file_name}' is:")
print(sha256_hash)