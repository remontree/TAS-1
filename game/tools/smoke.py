import json, ssl, sys, time, urllib.request, http.cookiejar
B = "https://127.0.0.1:28443"
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
def client():
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx), urllib.request.HTTPCookieProcessor(cj))
def call(op, path, body=None, hdr=True):
    req = urllib.request.Request(B + path, data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    if body is not None:
        req.add_header("Content-Type", "application/json")
        if hdr: req.add_header("X-TAS1", "1")
    try:
        with op.open(req, timeout=20) as r: return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"null")
a = client()
print("meta", call(a, "/api/meta")[1]["lut"])
print("me (anon)", call(a, "/api/me"))
print("csrf", call(a, "/api/register", {"name":"x"}, hdr=False))
reg = {"name":"테스트","student_id":"20260001","dept":"소프트웨어학과","phone":"010-1234-5678","consent":True}
print("register", call(a, "/api/register", reg))
print("dup", call(client(), "/api/register", reg)[0])
s, me = call(a, "/api/me"); print("me", s, me["masked"], me["phone"], me["official_left"])
s, row = call(a, "/api/engine/33333"); print("engine", s, row["isp"], row["impulse"], len(row["thrust"]))
t0=time.time(); s, r = call(a, "/api/launch", {"engine":"33333","air":{"body":3,"fin":2,"nose":1},"official":False}); print("practice", s, r["fl"]["outcome"], r["fl"]["apo"], r["score"], f"{(time.time()-t0)*1000:.0f}ms", len(r["fl"]["tr"]))
time.sleep(2.1); s, r = call(a, "/api/launch", {"engine":"10360","air":{"body":2,"fin":2,"nose":1},"official":True}); print("official", s, r["fl"]["outcome"], r["fl"]["apo"], r["score"], r["rank"], r["official_left"])
time.sleep(2.1); s, r = call(a, "/api/launch", {"engine":"33333","air":{"body":3,"fin":0,"nose":1},"official":True}); print("crash?", s, r["fl"]["outcome"], r["fl"].get("impactV"))
print("bad input", call(a, "/api/launch", {"engine":"99999","air":{"body":3,"fin":0,"nose":1}})[0])
print("nofit", call(a, "/api/launch", {"engine":"63363","air":{"body":0,"fin":2,"nose":1}}))
print("board", call(a, "/api/board?mode=target")[1]["rows"][:2])
f = call(a, "/api/feed")[1]; print("feed", f["stats"], [x["outcome"] for x in f["feed"]], f["rud"])
print("replay", call(a, f"/api/replay/{f['feed'][0]['id']}")[0])
b = client()
print("login wrong", call(b, "/api/login", {"student_id":"20260001","phone":"01000000000"}))
print("login ok", call(b, "/api/login", {"student_id":"20260001","phone":"01012345678"})[0], call(b, "/api/me")[1]["masked"])
c = client()
for i in range(6): s, j = call(c, "/api/login", {"student_id":"20260001","phone":"0100000000%d"%i})
print("lockout", s, j)
print("locked even w/ right phone", call(c, "/api/login", {"student_id":"20260001","phone":"01012345678"})[0])
print("admin no auth", call(c, "/api/admin/players")[0])
print("admin wrong", call(c, "/api/admin/login", {"password":"nope"})[0])
