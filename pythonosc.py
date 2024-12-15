from pythonosc import udp_client, dispatcher, osc_server
from threading import Thread
import time
import pickle
import torch
from dataLoad import rnn_model


# to install copy:
# pip install python-osc
# into terminal

# receiving data from diskliver MAX patch

# Callback function for received OSC messages
def osc_message_handler(address, *args):
    print(f"Received message: Address: {address} Args: {args}")

    # Pass data to the model
    processed_data = process_data_through_model(args)

    # Send the processed data to the client
    send_osc_message(client_ip, client_port, "/processed", processed_data)

# Load the trained model from model.pkl
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    model.eval()
    return model

# Process data through the RNN model
def process_data_through_model(data):
    # Convert the received data into a tensor
    input_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    # Initialize the hidden state for the model
    hidden = model.init_hidden()

    # Pass the input through the model
    with torch.no_grad():
        output, _ = model(input_tensor, hidden)

    # Convert the output to a list for sending back via OSC
    return output.squeeze(0).tolist()

# OSC Server to listen for incoming messages
def start_osc_server(ip, port):
    dispatcher_instance = dispatcher.Dispatcher()
    dispatcher_instance.map("/*", osc_message_handler)  # Catch-all handler

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher_instance)
    print(f"OSC Server is running on {ip}:{port}")
    server.serve_forever()  # Blocks the thread, so run it in a separate thread

# sending data created to MAX patch

# OSC Client to send messages
def send_osc_message(ip, port, address, value):
    client = udp_client.SimpleUDPClient(ip, port)
    client.send_message(address, value)
    #print(f"Message sent to {ip}:{port} at {address} with value: {value}")

if __name__ == "__main__":
    # Configuration
    client_ip = "10.17.244.147"  # Localhost IP address for testing
    client_port = 6070        # Port to send messages to MAX/MSP

    server_ip = "10.17.244.147"  # Localhost IP address for testing
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
