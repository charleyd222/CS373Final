from pythonosc import udp_client, dispatcher, osc_server
from threading import Thread
import time

# OSC Client to send messages
def send_osc_message(ip, port, address, value):
    client = udp_client.SimpleUDPClient(ip, port)
    client.send_message(address, value)
    #print(f"Message sent to {ip}:{port} at {address} with value: {value}")

# Callback function for received OSC messages
def osc_message_handler(address, *args):
    #print(f"Received message: Address: {address} Args: {args}")

# OSC Server to listen for incoming messages
def start_osc_server(ip, port):
    dispatcher_instance = dispatcher.Dispatcher()
    dispatcher_instance.map("/*", osc_message_handler)  # Catch-all handler

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher_instance)
    print(f"OSC Server is running on {ip}:{port}")
    server.serve_forever()  # Blocks the thread, so run it in a separate thread

if __name__ == "__main__":
    # Configuration
    client_ip = "10.17.244.147"  # Localhost for testing
    client_port = 6070        # Port to send messages to MAX/MSP

    server_ip = "10.17.244.147"  # Localhost for testing
    server_port = 9011        # Port to listen for messages from MAX/MSP

    # Start OSC server in a separate thread
    server_thread = Thread(target=start_osc_server, args=(server_ip, server_port))
    server_thread.daemon = True
    server_thread.start()

    # Allow some time for the server to start
    time.sleep(1)

    # Keep the program running to allow the server to receive messages
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")