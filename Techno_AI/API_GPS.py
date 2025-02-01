import requests

response = requests.get("http://ip-api.com/json/")
print(response.json())
