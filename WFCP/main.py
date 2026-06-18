from actors import Client, Server, Channel

# Create clients
clients = []
num_of_clients = 5

for k in range(num_of_clients):
    data_k = calibration_data_split[k]
    clients.append(Client(data_k, model, M))