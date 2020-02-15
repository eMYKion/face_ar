import argparse
import pandas as pd
import json
import socket

IP_SEND = "laptop-s6gp0saa.wifi.cmu.edu"
IP_PORT = 8000

def send_data(host_ip, port, data):

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("socket created successfully")
    except socket.error as err:
        print("socket creation failed:\n%s" % err)

    sock.connect((host_ip, port))
    sock.sendall(data.encode())
    #rec = sock.recv(1024)
    sock.close()
    #print("recieved ", repr(rec))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="send json info to a socket from a csv file")

    parser.add_argument("-f", action="store", dest="filename", type=str, required=True)
    parser.add_argument("-q", action="store", dest="query", nargs="+", required=True)

    args = parser.parse_args()
    
    query = ' '.join(args.query)
    
    df = pd.read_csv(args.filename)
    
    row = df.loc[df["name"] == query]

    json_data = {}
    for col in df.columns:
        #print(col, row[col][0])
        if col == "year":
            json_data.update({col:int(row[col][0])})
        else:
            json_data.update({col:str(row[col][0])})
            

    print(json_data)
    send_str = json.dumps(json_data)

    send_data(IP_SEND, IP_PORT, send_str)
    
    



    #print(row.to_dict())
