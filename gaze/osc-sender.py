from pythonosc.udp_client import SimpleUDPClient
from time import sleep

ip = "127.0.0.1"
port = 10000  # Must match TD's OSC In CHOP port
client = SimpleUDPClient(ip, port)

valueLabel = "/value"
arrayLabel = "/array"
stringLabel = "/string"



counter = 0
counterMax = 1000
interval = .5

while True:
    if counter > counterMax:
        counter = 0
        
    client.send_message(valueLabel, counter*.1)
    client.send_message(arrayLabel, [counter, counter+1, counter+2])
    client.send_message(stringLabel, "hello")
    counter = counter + 1
    sleep(interval)
