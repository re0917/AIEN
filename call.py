import requests
import time

def lineNotifyMessage(msg):

    headers = {
        "Authorization": "Bearer XmpnTDTLdaBULlfGO5FNBuOqKouDnpDh1cUIoeFuKAi" ,
        "Content-Type" : "application/x-www-form-urlencoded"
    }
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    payload = {'message': msg +t}
    r = requests.post("https://notify-api.line.me/api/notify", headers = headers, params = payload)
    
    return r.status_code

