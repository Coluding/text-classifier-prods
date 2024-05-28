import base64
from cryptography.fernet import Fernet
import hashlib

# Your custom 20-digit secret key
custom_key = 'fnienri2020203djdjd2pdoocnpd'

# Hash the custom key to ensure it is 32 bytes long
hashed_key = hashlib.sha256(custom_key.encode()).digest()

# Base64 encode the hashed key
base64_key = base64.urlsafe_b64encode(hashed_key[:32])

cipher_suite = Fernet(base64_key)

# Read the encrypted CSV file
with open('/home/lubi/Downloads/enc_data (1).csv', 'rb') as file:
    encrypted_data = file.read()

# Decrypt the data
decrypted_data = cipher_suite.decrypt(encrypted_data)

# Save the decrypted data to a new file
with open('training_data.csv', 'wb') as file:
    file.write(decrypted_data)

print("File decrypted successfully using custom key.")
