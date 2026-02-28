import socket
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for Hollow Knight HP data on {UDP_IP}:{UDP_PORT}...")
print("Make sure the game is running and the mod is installed!\n")

while True:
    data, addr = sock.recvfrom(1024)
    print(f"Received from Mod: {data.decode('utf-8')}")
