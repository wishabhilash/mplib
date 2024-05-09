from fyers_apiv3 import fyersModel
import webbrowser
import base64
import hmac
import os
import struct
import time
from urllib.parse import urlparse, parse_qs
import requests
import os
from retry import retry
from .shoonya import NorenApi
import pyotp


class BaseSession:
    def init_client(self):
        raise NotImplementedError
    

class FyersSession(BaseSession):
    def __init__(self, 
                 client_id=os.environ["FYERS_CLIENT_ID"],
                 secret_key=os.environ["FYERS_SECRET_KEY"],
                 username=os.environ["FYERS_USERNAME"],
                 totp_key=os.environ["FYERS_TOTP_KEY"],
                 pin=os.environ["FYERS_PIN"]
        ) -> None:
        self.redirect_uri = (
            "http://127.0.0.1:8080"  ## redircet_uri you entered while creating APP.
        )
        self.grant_type = "authorization_code"  ## The grant_type always has to be "authorization_code"
        self.response_type = "code"  ## The response_type always has to be "code"
        self.state = "sample"  ##  The state field here acts as a session manager. you will be sent with the state field after successfull generation of auth_code

        self.client_id = client_id
        self.secret_key = secret_key
        self.username = username
        self.totp_key = totp_key
        self.pin = pin

    def createSessionURL(self):
        appSession = fyersModel.SessionModel(
            self.client_id,
            self.redirect_uri,
            self.response_type,
            state=self.state,
            secret_key=self.secret_key,
            grant_type=self.grant_type,
        )

        # ## Make  a request to generate_authcode object this will return a login url which you need to open in your browser from where you can get the generated auth_code
        generateTokenUrl = appSession.generate_authcode()

        print((generateTokenUrl))
        webbrowser.open(generateTokenUrl, new=1)

    def totp(self, key, time_step=30, digits=6, digest="sha1"):
        key = base64.b32decode(key.upper() + "=" * ((8 - len(key)) % 8))
        counter = struct.pack(">Q", int(time.time() / time_step))
        mac = hmac.new(key, counter, digest).digest()
        offset = mac[-1] & 0x0F
        binary = struct.unpack(">L", mac[offset : offset + 4])[0] & 0x7FFFFFFF
        return str(binary)[-digits:].zfill(digits)

    def init_client(self):
        token = self._get_token()
        self.client = fyersModel.FyersModel(
            client_id=self.client_id, token=token, log_path=os.getcwd()
        )
        return self.client

    @retry(delay=2, tries=20)
    def _get_token(self):
        headers = {
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        }

        s = requests.Session()
        s.headers.update(headers)

        data1 = f'{{"fy_id":"{base64.b64encode(f"{self.username}".encode()).decode()}","app_id":"2"}}'
        r1 = s.post("https://api-t2.fyers.in/vagator/v2/send_login_otp_v2", data=data1)

        request_key = r1.json()["request_key"]
        data2 = f'{{"request_key":"{request_key}","otp":{self.totp(self.totp_key)}}}'
        r2 = s.post("https://api-t2.fyers.in/vagator/v2/verify_otp", data=data2)
        assert r2.status_code == 200, f"Error in r2:\n {r2.text}"

        request_key = r2.json()["request_key"]
        data3 = f'{{"request_key":"{request_key}","identity_type":"pin","identifier":"{base64.b64encode(f"{self.pin}".encode()).decode()}"}}'
        r3 = s.post("https://api-t2.fyers.in/vagator/v2/verify_pin_v2", data=data3)
        assert r3.status_code == 200, f"Error in r3:\n {r3.json()}"

        headers = {
            "authorization": f"Bearer {r3.json()['data']['access_token']}",
            "content-type": "application/json; charset=UTF-8",
        }
        data4 = f'{{"fyers_id":"{self.username}","app_id":"{self.client_id[:-4]}","redirect_uri":"{self.redirect_uri}","appType":"100","code_challenge":"","state":"abcdefg","scope":"","nonce":"","response_type":"code","create_cookie":true}}'
        r4 = s.post("https://api.fyers.in/api/v2/token", headers=headers, data=data4)
        assert r4.status_code == 308, f"Error in r4:\n {r4.json()}"

        parsed = urlparse(r4.json()["Url"])
        auth_code = parse_qs(parsed.query)["auth_code"][0]

        session = fyersModel.SessionModel(
            self.client_id,
            self.redirect_uri,
            self.response_type,
            secret_key=self.secret_key,
            grant_type=self.grant_type,
        )
        session.set_token(auth_code)
        response = session.generate_token()

        return response["access_token"]


class ShoonyaSession(BaseSession):
    def __init__(self,
                 client_id=os.environ["SHOONYA_CLIENT_ID"],
                 password=os.environ["SHOONYA_PASSWORD"],
                 secret_key=os.environ["SHOONYA_SECRET_KEY"],
                 totp_key=os.environ["SHOONYA_TOTP_KEY"],
                 ) -> None:
        self.client_id = client_id
        self.secret_key = secret_key
        self.password = password
        self.totp_key = totp_key

        self._api = NorenApi(
            host='https://api.shoonya.com/NorenWClientTP',
            websocket='wss://api.shoonya.com/NorenWSTP',
        )

    @retry(delay=2, tries=20)
    def init_client(self):
        res = self._api.login(
            self.client_id, 
            self.password, 
            pyotp.TOTP(self.totp_key).now(), 
            f'{self.client_id}_U',
            self.secret_key, 
            'abc1234'
        )
        
        self._api.set_session(self.client_id, self.password, res['susertoken'])
        return self._api