# gunicorn 설정 — server_config.json 을 읽어 HTTPS 로 바로 서비스한다
import json
import os
import ssl

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG = json.load(open(os.path.join(_BASE, "server_config.json"), encoding="utf-8"))

chdir = os.path.join(_BASE, "server")
wsgi_app = "app:app"
bind = f"{_CFG.get('host', '0.0.0.0')}:{_CFG['port']}"
workers = 1                      # 레이트 리밋·잠금 상태를 한 프로세스에서 공유
worker_class = "gthread"
threads = int(_CFG.get("threads", 16))
timeout = 30
graceful_timeout = 10
keepalive = 5
limit_request_line = 2048
limit_request_fields = 50
limit_request_field_size = 4096
accesslog = os.path.join(_BASE, "data", "logs", "access.log")
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(t)s "%(r)s" %(s)s %(b)s %(M)sms'
_tls = _CFG.get("tls", {})
if _tls.get("enabled", True):
    certfile = os.path.join(_BASE, _tls.get("cert", "certs/server.crt"))
    keyfile = os.path.join(_BASE, _tls.get("key", "certs/server.key"))
    ssl_version = ssl.PROTOCOL_TLS_SERVER
    ciphers = "ECDHE+AESGCM:ECDHE+CHACHA20"
